"""
Copyright (c) 2019-2021 Chao Zhang, Dengdong Fan, Zewen Wu, Kai Yang, Pengxiang Xu
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import shutil
import argparse
import torch

class MPIEnv:
    def __init__(self, world_size=None, rank=None, local_rank=None):
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        assert world_size > 0
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        assert (rank>=0) and (rank<world_size)
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        assert (local_rank>=0) and (local_rank<world_size)
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.is_master = (rank==0)
        self.is_distributed = (world_size > 1)
        self.is_local_rank0 = (local_rank==0)

    def __str__(self):
        ret = f'MPIEnv(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank})'
        return ret

    __repr__ = __str__


def get_parser_seng():
    parser = argparse.ArgumentParser(description='SENG optimizer')
    parser.add_argument('--datadir', default='/mnt/ImageNet2012', help='Place where data are stored')
    parser.add_argument('--logdir', default='log', type=str, help='where logs go')
    parser.add_argument('--lr', default=0.145, type=float, help='learning rate')
    parser.add_argument('--lr_warmup', default=0.01, type=float, help='learning rate warmup')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='batch size per gpu')
    parser.add_argument('--epoch', default=50, type=int, help='epoch')
    parser.add_argument('--epoch_end', default=60, type=int)
    parser.add_argument('--lr_exponent', default=6, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--lr_decay_rate', default=0.9, type=float, help='the rate damping is decayed')
    parser.add_argument('--warmup_epoch', default=5, type=int, help='first k epoch to gradually increase learning rate')
    # parser.add_argument('--final_tune_epoch', default=45, type=int, help='decrease learning rate rapidly at k epoch')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='label smoothing parameter')
    parser.add_argument('--damping', default=0.17, type=float, help='damping')
    parser.add_argument('--curvature_update_freq', default=500, type=int, help='The frequency to update inverse fisher matrix') #618
    parser.add_argument('--fim_col_sample_size', type=int, default=512, help='subsample count of col')
    parser.add_argument('--im_size_threshold', type=int, default=700000, help='only approximate over this size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--mixup', default=0., type=float, help='mixup interpolation coefficient')
    parser.add_argument('--local_rank', type=int, help='provided by torch.distributed.launch')
    parser.add_argument('--short-epoch', action='store_true', help='make epochs short (for debugging)')
    parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='log/print every this many steps')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode')
    parser.add_argument('--use-dali', action='store_false', help='use nvidia.dali gpu mode to do data loading work')
    return parser


def check_phase(lr_phase, dl_phase):
    assert all(('lr' in x) and ('bs' not in x) for x in lr_phase)
    assert all(('lr' not in x) and ('bs' in x) for x in dl_phase)

    assert all(len(x['lr'])==2 for x in lr_phase)
    assert all(len(x['ep'])==2 for x in lr_phase), 'linear learning rates must contain end epoch'
    assert all(x['ep'][0]<x['ep'][1] for x in lr_phase)
    assert lr_phase[0]['ep'][0]==0
    assert all(x['ep'][1]==y['ep'][0] for x,y in zip(lr_phase[:-1],lr_phase[1:]))
    for x in lr_phase:
        if 'type' not in lr_phase:
            x['type'] = 'linear'
    assert {x['type'] for x in lr_phase} <= {'linear','exp'}

    assert dl_phase[0]['ep']==0
    assert len({x['ep'] for x in dl_phase})==len(dl_phase)
    for x in dl_phase:
        assert 'val_bs' in x
        # tmp0 = {128:512, 224:256, 288:128}
        # x['val_bs'] = max(x['bs'], tmp0[x['sz']])
        if 'rect_val' not in x:
            x['rect_val'] = False
        if 'min_scale' not in x:
            x['min_scale'] = 0.08
    return lr_phase, dl_phase


def save_file_for_reproduce(save_list, logdir):
    assert os.path.abspath(save_list[0])==save_list[0], 'first element should be __file__'
    dirname = os.path.dirname(save_list[0])
    save_list = [save_list[0]] + [os.path.join(dirname,x) for x in save_list[1:]]
    for x in save_list:
        if os.path.exists(x):
            shutil.copy2(x, logdir)
        else:
            print(f'WARNING: file "{x}" not exists')


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing/(pred.shape[1]-1))
            true_dist.scatter_(1, target.view(-1,1), 1-self.smoothing)
        return torch.mean(torch.sum(-true_dist*pred, dim=1))


class LRScheduler:
    def __init__(self, optimizer, phase):
        self.optimizer = optimizer
        self.epoch_to_phase = {y:x for x in phase for y in range(*x['ep'])}
        self.lr = None
        self.lr_epoch_start = None
        self.lr_epoch_end = None

    def update_lr(self, epoch, ind_batch, batch_tot, exponent = 6,epoch_end = 60,lr_decay_rate=0.87,warmup_epoch=5):
        phase = self.epoch_to_phase[epoch]
        lr0, lr1 = phase['lr']
        ep0, ep1 = phase['ep']
        assert phase['type'] in {'linear','exp'}
        tmp0 = (epoch - ep0 + ind_batch/batch_tot) / (ep1 - ep0)
        if epoch< warmup_epoch:
            self.lr = lr0 + (lr1-lr0) * tmp0
        else: #exp
            self.lr = lr0 * (1- (epoch - ep0 + ind_batch/batch_tot)/epoch_end)**exponent
            if epoch > 39:
                self.lr = self.lr /10
            elif epoch > 41:
                self.lr = self.lr /10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        if epoch<warmup_epoch:
            self.lr_epoch_start = lr0 + (lr1-lr0) * tmp0
            self.lr_epoch_end   = lr0 + (lr1 - lr0) * (epoch - ep0 + 1+ ind_batch/batch_tot) / (ep1 - ep0)
        else:
            self.lr_epoch_start = lr0 * (1- (epoch - ep0 + ind_batch/batch_tot)/epoch_end)**exponent
            self.lr_epoch_end = lr0 * (1- (epoch+1 - ep0 + ind_batch/batch_tot)/epoch_end)**exponent


def my_topk(logits, label, topk=(1,)):
    prediction = logits.topk(max(topk), dim=1)[1]
    tmp0 = prediction==label.view(-1,1)
    ret = [tmp0[:,:x].sum().item() for x in topk]
    return ret


def sum_tensor(tensor):
    ret = tensor.clone()
    torch.distributed.all_reduce(ret, op=torch.distributed.ReduceOp.SUM)
    return ret
