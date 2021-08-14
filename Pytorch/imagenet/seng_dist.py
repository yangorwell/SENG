# An implementation of SENG optimizer from:
#
# Sketchy Empirical Natural Gradient Methods for Deep Learning 
# Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu
# Preprint Paper: https://arxiv.org/pdf/2006.05924.pdf
# contact: yangminghan at pku.edu.cn, taroxd at pku.edu.cn, wenzw at pku.edu.cn

# Copyright (c) 2021 Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

# FIM: Fisher Information Matrix

SUPPORTED_LAYER_TYPE = {'Conv2dEx', 'LinearEx'}

def get_group_gadient_vector(group):
    glist = [x.grad.contiguous() for x in group['params']]
    param = glist[0].detach()
    if group['layer_type'] == 'Conv2dEx':
        param = param.view(param.shape[0], -1)
    elif group['layer_type'] == 'BatchNorm2dEx':
        param = param.view(-1,1)
    if len(glist)>1:
        b = glist[1].detach()
        if group['layer_type'] in SUPPORTED_LAYER_TYPE:
            param = torch.cat([param, b.view(-1, 1)], dim=1)
        else:
            param = torch.cat([param.view(-1), b.view(-1)])
    ret = param.view(-1)
    return ret


def precond(fim_invmat, weight_grad, bias_grad, layer_type, state, damping, world_size=1):
    g = weight_grad
    s = g.shape
    sync = 0
    if layer_type == 'Conv2dEx':
        g = g.contiguous().view(s[0], s[1]*s[2]*s[3]) #(out_channel,in_channel*kernel_size)
    elif layer_type == 'BatchNorm2dEx':
        g = g.unsqueeze(1) #(out_channel,1)
    if bias_grad is not None:
        g = torch.cat([g, bias_grad.view(-1,1)], dim=1)
    orig_shape = g.shape
    g = g.view(-1)

    g.div_(damping)
    if 'sample_index' in state:
        sample_index = state['sample_index']
        g_sub = g[sample_index] * (g.shape[0] / sample_index.shape[0])
    else:
        g_sub = g
    if 'dw' in state:
        tmp0 = fim_invmat @ (state['dw_sub'] @ g_sub) #(batch_size,)
        # dw (batch_size,out_channel*in_channel)
        lambdaUb = torch.mv(state['dw'].t(), tmp0)
        sync = 1
    else:
        gy_sub = state['sketch_gy_sub']
        x_sub = state['sketch_x_sub']
        gy = state['sketch_gy']
        x = state['sketch_x']
        mat_v = g_sub.view(gy_sub.shape[1], x_sub.shape[1])
        tmp0 = fim_invmat @ (((mat_v @ x_sub) * gy_sub).sum((1, 2))) #(batch_size,)
        batch_size = tmp0.shape[0]
        coeff_gy = tmp0.abs().sqrt_()
        gy_v = (gy.reshape(batch_size, -1).t() @ coeff_gy).reshape(gy.shape[1], gy.shape[2])
        tmp1 = torch.max(coeff_gy.sum(), torch.tensor([1e-5], dtype=coeff_gy.dtype, device=coeff_gy.device)[0]) #TODO torch.maximum
        x_v = (x.view(batch_size,-1).t() @ (tmp0/tmp1)).view(x.shape[1], x.shape[2])
        if world_size>1:
            with torch.no_grad():
                gy_v.div_(world_size)
                x_v.div_(world_size)
                dist.all_reduce(gy_v, dist.ReduceOp.SUM, async_op=False)
                dist.all_reduce(x_v, dist.ReduceOp.SUM, async_op=False)
        lambdaUb = torch.mm(gy_v, x_v.t()).view(-1)
    g.sub_(lambdaUb)

    g = g.view(orig_shape)
    if bias_grad is not None:
        gb = g[:, -1].contiguous().view(*bias_grad.shape)
        g = g[:, :-1]
    else:
        gb = None
    g = g.contiguous().view(*s)
    return g, gb,sync


class SENG(torch.optim.Optimizer):
    def __init__(self, net, damping, update_freq=1, im_size_threshold=700000, col_sample_size=512, loss_scaler=None, world_size=1):
        """
        net (torch.nn.Module): Network to precondition.
        damping (float): damping, the `lambda` in the paper.
        update_freq (int): how often should the curvature matrix be updated.
        im_size_threshold (int): the threshold of paramter size. When exceeded, use implicit version of (E)FIM.
        col_sample_size (int): Sample size to the rows of matrices G and A. None if no sketching.
        """
        self.fp16 = loss_scaler is not None
        self.loss_scaler = loss_scaler
        self.damping = damping
        self.update_freq = update_freq
        self.iteration_counter = 0
        self.svd_rank = 256
        self.im_size_threshold = im_size_threshold
        self.col_sample_size = col_sample_size
        self.world_size = world_size
        param_list = []
        for x in net.modules():
            if hasattr(x, 'weight'):
                tmp0 = [x.weight,x.bias] if x.bias is not None else [x.weight]
                param_list.append({'params':tmp0, 'mod':x, 'layer_type':x.__class__.__name__})
        super().__init__(param_list, {})
        assert sum(y.numel() for x in self.param_groups for y in x['params']) == sum(x.numel() for x in net.parameters())
        supported_group_count = 0
        for x in self.param_groups:
            state = self.state[x['params'][0]]
            if x['layer_type'] in SUPPORTED_LAYER_TYPE:
                state['group_id'] = supported_group_count
                supported_group_count += 1
        self.state['fim_invmats'] = [None for _ in range(supported_group_count)]
        self.device = self.param_groups[0]['params'][0].device

    def step(self):
        if self.iteration_counter%self.update_freq==0:
            for x in self.param_groups:
                if x['layer_type'] in SUPPORTED_LAYER_TYPE:
                    self._compute_fim(x)
                    # TODO assert fim is not NAN
            # len(param_groups) = 107
            # len(sample_GA) = 10
            # len(sample_dw) = 8

        dist_handles = []
        for group in self.param_groups:
            if group['layer_type'] in SUPPORTED_LAYER_TYPE:
                if len(group['params']) == 2:
                    weight, bias = group['params']
                    bias_grad = bias.grad.detach()
                else:
                    weight = group['params'][0]
                    bias = None
                    bias_grad = None
                state = self.state[weight]
                fim_invmat = self.state['fim_invmats'][state['group_id']]
                gw,gb,sync = precond(fim_invmat, weight.grad.detach(), bias_grad, group['layer_type'], state, self.damping, self.world_size)
                weight.grad.data = gw
                if bias is not None:
                    bias.grad.data = gb
                if self.world_size>1 and sync:
                    with torch.no_grad():
                        weight.grad.div_(self.world_size)
                        dist_handles.append(dist.all_reduce(weight.grad, dist.ReduceOp.SUM, async_op=True))
                    if bias is not None:
                        with torch.no_grad():
                            bias.grad.div_(self.world_size)
                            dist_handles.append(dist.all_reduce(bias.grad, dist.ReduceOp.SUM, async_op=True))
            else:
                for x in group['params']:
                    x.grad.data.div_(1.0 + self.damping)
        for x in dist_handles:
            x.wait()

        self.iteration_counter += 1

    def _compute_fim(self, group):
        mod = group['mod']
        params = group['params']
        state = self.state[params[0]]
        x = mod.last_input.detach()
        gy = mod.last_output.grad.detach()
        if self.fp16:
            assert (x.dtype==torch.float16) and (gy.dtype==torch.float16)
            x = x.float()
            gy = gy.float() / self.loss_scaler.get_scale()
        del mod.last_input
        del mod.last_output
        batch_size = gy.shape[0]
        gy.mul_(batch_size**0.5)

        if (group['layer_type']=='LinearEx') or (group['layer_type']=='Conv2dEx'):
            if group['layer_type']=='Conv2dEx':
                x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride, dilation=mod.dilation) #(batch_size,in_channel*kernel_size,?)
                gy = gy.view(batch_size, gy.shape[1], -1) #(batch_size,out_channel,?)
            if group['layer_type'] == 'LinearEx':
                x = x.unsqueeze(2) #(batch_size,in_channel,1)
                gy = gy.unsqueeze(2) #(batch_size,out_channel,1)
            if len(params)>1:
                x = torch.cat([x, torch.ones_like(x[:,:1])], dim=1)

            dw_len = gy.shape[1] * x.shape[1]
            if dw_len <= self.im_size_threshold:
                dw = torch.bmm(gy, x.transpose(1, 2))
                assert dw.dtype==torch.float32
                # conv2d(batch_size,out_channel,in_channel*kernel_size+1)
                # linear(batch_size,out_channel,in_channel)
            else:
                if x.shape[2] > self.svd_rank:
                    u, s, v = torch.svd(gy, compute_uv=True)
                    gy = u[:,:,:self.svd_rank] * s[:,:self.svd_rank].unsqueeze(1)
                    x = torch.matmul(x, v[:,:,:self.svd_rank])
                p = x.shape[2]

                col_sample_size = self.col_sample_size
                assert col_sample_size < max(x.size(1), gy.size(1)) #is_sample
                dw_all_index = torch.arange(dw_len, device=self.device).view(gy.shape[1], x.shape[1])
                N0 = gy.shape[1]
                if col_sample_size < N0:
                    ind0 = np.sort(np.random.choice(N0, col_sample_size, replace=False))
                    dw_all_index = dw_all_index[ind0]
                    gy_sub = gy[:,ind0] * (N0/col_sample_size)
                else:
                    gy_sub = gy
                N0 = x.shape[1]
                if col_sample_size < N0:
                    ind0 = np.sort(np.random.choice(N0, col_sample_size, replace=False))
                    dw_all_index = dw_all_index[:,ind0]
                    x_sub = x[:,ind0] * (N0/col_sample_size)
                else:
                    x_sub = x

                x_bat = x_sub.permute(1, 0, 2).contiguous().view(x_sub.shape[1], -1) #(in_channel,batch_size*?)
                gy_bat = gy_sub.permute(1, 0, 2).contiguous().view(gy_sub.shape[1], -1) #(out_channel,batch_size*?)
                tmp0 = (x_bat.t() @ x_bat) * (gy_bat.t() @ gy_bat)
                rows, cols = tmp0.shape
                tmp1 = tmp0.view(rows // p, p, cols // p, p).sum((1, 3))
                tmp1.diagonal().add_(self.damping)
                self.state['fim_invmats'][state['group_id']] = torch.inverse(tmp1)
                state['sketch_gy'] = gy
                state['sketch_x'] = x
                state['sketch_gy_sub'] = gy_sub
                state['sketch_x_sub'] = x_sub
                state['sample_index'] = dw_all_index.view(-1)
                return
        else:
            assert group['layer_type'] == 'BatchNorm2dEx'
            tmp0 = F.batch_norm(x, None, None, None, None, True, 0.0, mod.eps) #TODO strange where is running_mean
            dw = torch.cat([(tmp0*gy).sum([2,3]).unsqueeze(2), gy.sum([2,3]).unsqueeze(2)], dim=2)
            # (batch_size,out_channel,2)

        dw = dw.view(batch_size, -1)
        col_sample_size = self.col_sample_size * self.col_sample_size
        dw_len = dw.shape[1]
        is_sample = col_sample_size < dw_len
        if is_sample:
            tmp0 = np.sort(np.random.choice(dw_len, col_sample_size, replace=False))
            state['sample_index'] = torch.tensor(tmp0, device=self.device)
            dw_sub = dw[:,state['sample_index']] * (dw_len / col_sample_size)
        else:
            dw_sub = dw
        tmp0 = torch.matmul(dw_sub, dw_sub.t())
        tmp0.diagonal().add_(self.damping)
        self.state['fim_invmats'][state['group_id']] = torch.inverse(tmp0)
        state['dw_sub'] = dw_sub
        state['dw'] = dw #(batch_size,out_channel*in_channel)
