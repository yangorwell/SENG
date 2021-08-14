# SENG
# Copyright (c) 2021 Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This file exports an optimizer class called `SENG`.
#
# Here is a typical example of incorporating SENG to your program.
#
# ```
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)
# preconditioner = SENG(net, 0.8, update_freq=50)
#
# for inputs, targets in trainloader:
#    output = net(inputs)
#    loss = criterion(outputs, targets)
#    optimizer.zero_grad()
#    loss.backward()

#    preconditioner.step()
#    optimizer.step()
# ```

import time
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

class SENG(Optimizer):
    def __init__(self, net, damping, update_freq=1, verbose=False, subsample=None,
        im_size_threshold=1000000, col_sample_size=256):
        """SENG preconditioner

        Args:
            net (torch.nn.Module): Network to precondition.
            damping (float): damping, the `lambda` in the paper.
            update_freq (int): how often should the curvature matrix be updated.
            verbose (bool): print additional information.
            subsample (int or None): use less samples to compute curvature matrix.
                None if no subsampling.
            im_size_threshold (int): the threshold of paramter size.
                When exceeded, use implicit version of (E)FIM.
            col_sample_size (int or None): Sample size to the rows of matrices G and A.
                None if no sketching.
        """

        self.damping = damping
        self.update_freq = update_freq
        self.params = []
        self.subsample = subsample
        self.verbose = verbose
        self.iteration_counter = 0
        self.svd_rank = 16
        self.im_size_threshold = im_size_threshold
        self.net = net
        for mod in net.modules():
            if hasattr(mod, 'weight'):
                mod_class = mod.__class__.__name__
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super().__init__(self.params, {})
        self.total_numel = 0
        self._supported_group_count = 0
        for group in self.param_groups:
            params = group['params']
            state = self.state[params[0]]
            state['offset_start'] = self.total_numel
            self.total_numel += sum((v.numel() for v in params))
            state['offset_end'] = self.total_numel
            weight = params[0]
            if self._is_supported_group(group):
                state['group_id'] = self._supported_group_count
                self._supported_group_count += 1
        assert self.total_numel == sum((v.numel() for v in net.parameters()))
        self.dtype = weight.dtype
        self.device = weight.device
        if col_sample_size is None:
            col_sample_size = 1e50
        self.col_sample_size = col_sample_size
        if self.verbose:
            self.gbuffer = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
            self.hgbuffer = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)

    def step(self, inv_fim=None):
        """Performs one step of preconditioning.
            inv_fim (bool): set to True to inv fim. Default to (self.iteration_counter % self.update_freq)
        """

        # g is for logging only
        if self.verbose:
            g = self.gbuffer
            for group in self.param_groups:
                state = self.state[group['params'][0]]
                s1 = state['offset_start']
                s2 = state['offset_end']
                layer_g = self._get_param_and_grad(group, clone=False, gradonly=True)
                g[s1:s2] = layer_g

        do_inv_fim = self.iteration_counter % self.update_freq == 0 if inv_fim is None else inv_fim

        if do_inv_fim:
            self.compute_all_fim_invs()

        for group in self.param_groups:
            if self._is_supported_group(group):
                # Getting parameters
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None
                state = self.state[weight]
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                # Updating gradients
                weight.grad.data = gw
                if bias is not None:
                    bias.grad.data = gb
            else:
                for param in group['params']:
                    param.grad.data.div_(1.0 + self.damping)

        if self.verbose:
            hg = self.hgbuffer
            for group in self.param_groups:
                state = self.state[group['params'][0]]
                s1 = state['offset_start']
                s2 = state['offset_end']
                layer_hg = self._get_param_and_grad(group, clone=False, gradonly=True)
                hg[s1:s2] = layer_hg

            gnorm2 = g.dot(g)
            hgnorm2 = hg.dot(hg)
            gtd = -(g.dot(hg))
            self.state['adg'] = gtd * (gnorm2 * hgnorm2)**(-0.5)
            self.state['normg'] = gnorm2**0.5
            self.state['normd'] = hgnorm2**0.5
        self.iteration_counter += 1

    def collect_param_and_grad(self):
        param = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)
        g = torch.empty(self.total_numel, device=self.device, dtype=self.dtype)

        for group in self.param_groups:
            state = self.state[group['params'][0]]
            s1 = state['offset_start']
            s2 = state['offset_end']
            layer_param, layer_g = self._get_param_and_grad(group, clone=False)
            param[s1:s2] = layer_param
            g[s1:s2] = layer_g
        return param, g

    def unprecond_all(self, v, inplace=False):
        result = v
        if not inplace:
            result = result.clone()
        for group in self.param_groups:
            state = self.state[group['params'][0]]
            s1 = state['offset_start']
            s2 = state['offset_end']
            layer_v = v[s1:s2]
            if self._is_supported_group(group):
                dw = state['dw']
                # v = v * damping + Ut @ U @ v
                result[s1:s2].addmv_(dw.t(), dw @ layer_v, beta=self.damping)
            else:
                result[s1:s2].mul_(1.0 + self.damping)
        return result

    def compute_all_fim_invs(self):
        for group in self._supported_group():
            self._compute_fim(group)
        self.state['fim_invmats'] = torch.inverse(self.state['fim_invmats'])


    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        fim_invmat = self.state['fim_invmats'][state['group_id'], :, :]
        tmpbuff = state.get('tmpbuff')

        g = weight.grad.detach()
        s = g.shape
        if group['layer_type'] == 'Conv2dEx':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        elif group['layer_type'] == 'BatchNorm2dEx':
            g = g.unsqueeze(1)
        if bias is not None:
            gb = bias.grad.detach()
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        orig_shape = g.size()
        g = g.view(-1)

        # F = (kI + U'U)
        # F^{-1}g = g/k - U' (kI+UU')^{-1} U g/k
        g.div_(self.damping)
        # tmp = dw @ g
        # tmp = fim_invmat @ tmp
        # tmp = dw.t() @ tmp
        tmp = self._dw_vector_product(state, g)
        tmp = fim_invmat @ tmp
        tmp = self._dwt_vector_product(state, tmp, out=tmpbuff)
        state['tmpbuff'] = tmp
        g.sub_(tmp)

        # split to weight and bias parts
        g = g.view(orig_shape)
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _compute_fim(self, group):
        """Approximate the inverse of FIM"""

        # Compute gradient of each sample
        mod = group['mod']
        params = group['params']
        state = self.state[params[0]]

        dw = state.get('dw')   # used as buffer

        x = mod.last_input.detach()
        gy = mod.last_output.grad.detach()

        # free memory
        del mod.last_input
        del mod.last_output

        if self.subsample:
            x = x[:self.subsample]
            gy = gy[:self.subsample]
        batch_size = gy.size(0)
        whole_batch_size = batch_size
        if batch_size !=  256:
            self.iteration_counter -= 1
            print('bs',batch_size)
            return 
        if 'fim_invmats' not in self.state:
            self.state['fim_invmats'] = torch.empty((self._supported_group_count, batch_size, batch_size), dtype=self.dtype, device=self.device)

        # gy *= batch_size
        gy.mul_(batch_size / whole_batch_size**0.5)
        has_bias = len(params) == 2

        if group['layer_type'] == 'LinearEx' or group['layer_type'] == 'Conv2dEx':
            if group['layer_type'] == 'Conv2dEx':
                x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride, dilation=mod.dilation)
                gy = gy.view(gy.size(0), gy.size(1), -1)
            if has_bias:
                ones = torch.ones_like(x[:, :1])
                x = torch.cat([x, ones], dim=1)
            if group['layer_type'] == 'LinearEx':
                gy = gy.unsqueeze(2)
                x = x.unsqueeze(2)

            dw_len = gy.size(1) * x.size(1)
            if dw_len <= self.im_size_threshold:
                dw = torch.bmm(gy, x.transpose(1, 2), out=dw)
            else:
                gy, x = self._svd_approximate(gy, x, self.svd_rank)
                p = x.size(2)  # can be smaller than svd_rank

                col_sample_size = self.col_sample_size

                is_sample = col_sample_size < max(x.size(1), gy.size(1))
                state['is_sample'] = is_sample
                if is_sample:
                    if 'arange_buffer' not in state:
                        state['arange_buffer'] = torch.arange(dw_len, device=self.device)
                        state['sample_index'] = torch.empty(col_sample_size, device=self.device, dtype=torch.long)
                    dw_all_index = state['arange_buffer'].view(gy.size(1), x.size(1))
                    sample_index = state['sample_index']

                def make_sample(mat, dw_dim):
                    nonlocal dw_all_index
                    dimsize = mat.size(1)
                    if col_sample_size < dimsize:
                        mul_ratio = dimsize / col_sample_size
                        np_index = np.random.choice(dimsize, col_sample_size, replace=False)
                        mat_sample_index = sample_index[:col_sample_size]
                        mat_sample_index.copy_(torch.from_numpy(np_index))
                        dw_all_index = torch.index_select(dw_all_index, dim=dw_dim, index=mat_sample_index)
                        return torch.index_select(mat, dim=1, index=mat_sample_index).mul_(mul_ratio)
                    else:
                        return mat

                gy_sub = make_sample(gy, 0)
                x_sub = make_sample(x, 1)

                x_bat = x_sub.permute(1, 0, 2).contiguous().view(x_sub.size(1), -1)
                gy_bat = gy_sub.permute(1, 0, 2).contiguous().view(gy_sub.size(1), -1)
                fim_invmat = self._blocksum((x_bat.t() @ x_bat) * (gy_bat.t() @ gy_bat), p)

                state['sketch_gy'] = gy
                state['sketch_x'] = x
                state['sketch_gy_sub'] = gy_sub
                state['sketch_x_sub'] = x_sub
                if is_sample:
                    state['sample_index'] = dw_all_index.view(-1)
        elif group['layer_type'] == 'BatchNorm2dEx':
            no_affine_out = F.batch_norm(x, None, None, None, None, True, 0.0, mod.eps)
            dweight = (no_affine_out * gy).sum([2, 3])
            dbias = gy.sum([2, 3])
            dw = torch.cat([dweight.unsqueeze(2), dbias.unsqueeze(2)], dim=2, out=dw)
        else:
            raise TypeError("Unsupported type. Maybe bug")

        if dw is not None:
            dw = dw.view(batch_size, -1)
            # (kI + WW')^{-1}
            wwt_buff = self.state['fim_invmats'][state['group_id'], :, :]

            col_sample_size = self.col_sample_size * self.col_sample_size
            dw_len = dw.size(1)
            state['dw_len'] = dw_len
            is_sample = col_sample_size < dw_len
            if is_sample:
                if 'sample_index' not in state:
                    state['sample_index'] = torch.empty(col_sample_size, device=self.device, dtype=torch.long)
                sample_index = state['sample_index']
                mul_ratio = dw_len / col_sample_size
                np_index = np.random.choice(dw_len, col_sample_size, replace=False)
                sample_index.copy_(torch.from_numpy(np_index))
                dw_sub = state.get('dw_subbuff')
                dw_sub = torch.index_select(dw, dim=1, index=sample_index, out=dw_sub)
                dw_sub.mul_(mul_ratio)
            else:
                dw_sub = dw
            fim_invmat = torch.matmul(dw_sub, dw_sub.t(), out=wwt_buff)
            state['is_sample'] = is_sample
            state['dw_sub'] = dw_sub
            state['dw'] = dw
        fim_invmat.diagonal().add_(self.damping)
        self.state['fim_invmats'][state['group_id'], :, :] = fim_invmat

    def _svd_approximate(self, gy, x, p):
        if x.size(2) <= p:
            return gy, x
        ug, sg, vg = torch.svd(gy, some=True, compute_uv=True)
        ug = ug[:, :, :p]
        vg = vg[:, :, :p]
        sg_expand = sg[:, :p].unsqueeze(1).expand(*ug.size())
        return ug * sg_expand, x @ vg

    def _blocksum(self, mat, p):
        rows, cols = mat.size()
        assert(rows % p == 0 and cols % p == 0)
        return mat.view(rows // p, p, cols // p, p).sum((1, 3))

    def _dw_vector_product(self, state, v):
        if state['is_sample']:
            sample_index = state['sample_index']
            mul_ratio = v.size(0) / sample_index.size(0)
            v_sub = state.get('v_subbuff')
            v_sub = torch.index_select(v, dim=0, index=sample_index, out=v_sub)
            v_sub.mul_(mul_ratio)
            state['v_subbuff'] = v_sub
        else:
            v_sub = v
        if 'dw' in state:
            return state['dw_sub'] @ v_sub

        gy = state['sketch_gy_sub']
        x = state['sketch_x_sub']
        mat_v = v_sub.view(gy.size(1), x.size(1))
        # -- equivalent to --
        # wv2 = torch.zeros_like(wv)
        # for i in range(batch_size):
        #     for j in range(p):
        #         aij = x[i, :, j]
        #         gij = gy[i, :, j]
        #         wv2[i] += gij.t() @ mat_v @ aij
        return ((mat_v @ x) * gy).sum((1, 2))

    def _dwt_vector_product(self, state, v, out=None):
        if 'dw' in state:
             return torch.mv(state['dw'].t(), v, out=out)
        gy = state['sketch_gy']
        x = state['sketch_x']

        bs = v.size(0)
        v_sqrt = v.abs().sqrt_()
        coeff_gy = v_sqrt
        gy_v = (gy.reshape(bs, -1).t() @ coeff_gy).reshape(gy.size(1), gy.size(2))
        coeff_a = torch.div(v, coeff_gy.sum(), out=v_sqrt)
        x_v = (x.reshape(bs, -1).t() @ coeff_a).reshape(x.size(1), x.size(2))

        return torch.mm(gy_v, x_v.t(), out=out).view(-1)

        # result = torch.zeros((dw_len,), dtype=self.dtype, device=self.device, out=out)
        # mat_res = result.view(gy.size(1), x.size(1))
        # for i in range(v.size(0)):
        #     mat_res.addmm_(gy[i, :, :], x[i, :, :].t(), alpha=v[i])
        # return result

    def _supported_group(self):
        for group in self.param_groups:
            if self._is_supported_group(group):
                yield group

    def _is_supported_group(self, group):
        return (group['layer_type'] == 'Conv2dEx' or
            group['layer_type'] == 'LinearEx' or
            group['layer_type'] == 'BatchNorm2dEx')

    def _get_param_and_grad(self, group, clone=True, detach=True, gradonly=False):
        """Get gradient from weight and bias, and view as a vector."""
        params = group['params']
        glist = [param.grad.contiguous() for param in params]
        g = self._flatten_layer_param(group, glist, clone=clone, detach=detach)
        if gradonly:
            return g
        param = self._flatten_layer_param(group, params, clone=clone, detach=detach)
        return param, g

    def _flatten_layer_param(self, group, params, clone=True, detach=True):
        """Get gradient from weight and bias, and view as a vector."""
        if len(params) == 2:
            weight, bias = params
        else:
            weight = params[0]
            bias = None
        s = weight.shape
        param = weight
        if detach:
            param = param.detach()
        if group['layer_type'] == 'Conv2dEx':
            param = param.view(s[0], s[1]*s[2]*s[3])
        elif group['layer_type'] == 'BatchNorm2dEx':
            param = param.unsqueeze(1)
        if bias is None:
            if clone:
                param = param.clone()
        else:
            b = bias
            if detach:
                b = b.detach()
            if self._is_supported_group(group):
                param = torch.cat([param, b.view(-1, 1)], dim=1)
            else:
                param = torch.cat([param.view(-1), b.view(-1)])
        return param.view(-1)
