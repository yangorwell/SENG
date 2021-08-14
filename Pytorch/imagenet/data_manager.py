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
import gc
import torch
import torchvision
import numpy as np
import warnings
import imagesize #pip install imagesize
import glob

from dali_pipe import HybridTrainPipe, DaliIteratorGPU
from nvidia_dali_utils2 import PaddedShuffleDALIDataLoader, PaddedNoShuffleDALIDataLoader

# fast_collate()/torch.from_numpy()
warnings.filterwarnings("ignore", ("The given NumPy array is not writeable,"
        " and PyTorch does not support non-writeable tensors"), UserWarning)


class DataManager:

    def __init__(self, phases, ilsvrc_root, workers, use_dali, fp16, world_size=1, rank=0, local_rank=0):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.traindir = os.path.join(ilsvrc_root, 'train')
        self.valdir = os.path.join(ilsvrc_root, 'val')
        self.workers = workers
        self.fp16 = fp16
        self.epoch_to_phase = {x['ep']:x for x in phases}

        self.use_dali = use_dali
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.index_height_width = [(x,*imagesize.get(y)) for x,y in enumerate(sorted(glob.glob(f'{self.valdir}/*/*.JPEG')))]

    def set_epoch(self, epoch):
        if epoch in self.epoch_to_phase:
            self.current_phase = self.epoch_to_phase[epoch]

            sz = self.current_phase['sz']
            bs = self.current_phase['bs']
            val_bs = self.current_phase['val_bs']
            min_scale = self.current_phase['min_scale']
            rect_val = self.current_phase['rect_val']

            if self.use_dali:
                # train_loader = PaddedShuffleDALIDataLoader(self.traindir, batch_size=bs, crop=sz, min_crop_size=min_scale, fp16=self.fp16,
                #             num_worker=self.workers, seed=epoch, world_size=self.world_size, rank=self.rank, local_rank=self.local_rank)
                # train_sampler = None

                train_pipe = HybridTrainPipe(batch_size=bs, num_threads=self.workers, device_id=self.local_rank,
                            data_dir=self.traindir, crop=sz, dali_cpu=False, shuffle=True, mean=self.mean,
                            std=self.std, fp16=self.fp16, min_crop_size=min_scale, rank=self.rank, world_size=self.world_size)
                train_pipe.build()
                train_loader = DaliIteratorGPU(pipelines=train_pipe, mean=self.mean, std=self.std, fp16=self.fp16, pin_memory=True)
                train_sampler = None

                # val_loader = PaddedNoShuffleDALIDataLoader(self.valdir, batch_size=val_bs, crop=sz, rectangular=rect_val, fp16=self.fp16,
                #             num_worker=self.workers, world_size=self.world_size, rank=self.rank, local_rank=self.local_rank)
                # val_sampler = None
            else:
                tmp0 = torchvision.transforms.Compose([
                        torchvision.transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
                        torchvision.transforms.RandomHorizontalFlip()
                ])
                train_dataset = torchvision.datasets.ImageFolder(self.traindir, transform=tmp0)
                if self.world_size>1:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
                else:
                    train_sampler = None
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=(train_sampler is None),
                        num_workers=self.workers, pin_memory=True, collate_fn=fast_collate, sampler=train_sampler)
                train_loader = BatchTransformDataLoader(train_loader, fp16=self.fp16)

            val_dataset, val_sampler = my_create_validation_set(self.valdir, val_bs, sz, rect_val=rect_val,
                    index_height_width=self.index_height_width , world_size=self.world_size, rank=self.rank)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                num_workers=self.workers, pin_memory=True, collate_fn=fast_collate, batch_sampler=val_sampler)
            val_loader = BatchTransformDataLoader(val_loader, fp16=self.fp16)

            self.trn_dl = train_loader
            self.val_dl = val_loader
            self.trn_smp = train_sampler
            self.val_smp = val_sampler

            gc.collect() #clear memory before training, TODO is this necessary
        if hasattr(self.trn_smp, 'set_epoch'):
            self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'):
            self.val_smp.set_epoch(epoch)


class BatchTransformDataLoader():
    # Mean normalization on batch level instead of individual
    # https://github.com/NVIDIA/apex/blob/59bf7d139e20fb4fa54b09c6592a2ff862f3ac7f/examples/imagenet/main.py#L222
    # TODO changed in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L264
    def __init__(self, loader, fp16=True):
        self.loader = loader
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.fp16 = fp16
        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __len__(self):
        return len(self.loader)

    def process_tensors(self, image, label, non_blocking=True):
        assert image.ndim==4
        image = image.cuda(non_blocking=non_blocking)
        if self.fp16:
            image = image.half()
        else:
            image = image.float()
        image = image.sub_(self.mean).div_(self.std)
        label = label.cuda(non_blocking=non_blocking)
        return image,label

    def __iter__(self):
        return (self.process_tensors(x,y) for x,y in self.loader)


def fast_collate(batch_data):
    data_list = [x[0] for x in batch_data] #(list,PIL.image)
    label = torch.tensor([x[1] for x in batch_data], dtype=torch.int64)
    N0 = len(data_list)
    N1,N2 = data_list[0].height,data_list[0].width #TODO strange
    # N1,N2 = data_list[0].size
    # torch.channels_last or torch.contiguous_format, TODO apex example use channels_last for performance
    data = torch.empty((N0,3,N1,N2), dtype=torch.uint8).contiguous(memory_format=torch.contiguous_format)
    for ind0, data_i in enumerate(data_list):
        tmp0 = np.asarray(data_i, dtype=np.uint8)
        data[ind0] = torch.from_numpy(np.moveaxis(tmp0, 2, 0))
    return data, label


class CropArTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar = idx2ar
        self.target_size = target_size
    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1:
            w = int(self.target_size/target_ar)
            size = (w//8*8, self.target_size)
        else:
            h = int(self.target_size*target_ar)
            size = (self.target_size, h//8*8)
        return torchvision.transforms.functional.center_crop(img, size)

class ValDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, CropArTfm):
                    sample = tfm(sample, index)
                else:
                    sample = tfm(sample)
        return sample, target


class MySimpleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_index):
        self.batch_index = batch_index
    def __len__(self):
        return len(self.batch_index)
    def set_epoch(self, epoch):
        pass
    def __iter__(self):
        yield from self.batch_index

# DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images)
def my_split_valdataset(index_list, batch_size, world_size=1, rank=0):
    hf_ceil = lambda x,y: (x-1)//y + 1
    if world_size > 1:
        assert batch_size > 1 #could lead to len(batch)==0
        num_batch = hf_ceil(len(index_list), batch_size*world_size)
        batch_index = []
        for x in range(len(index_list)//(batch_size*world_size)):
            batch_index.append(index_list[((x*world_size+rank)*batch_size):((x*world_size+rank+1)*batch_size)])
        # separate last batch
        tmp0 = len(index_list)%(batch_size*world_size)
        if tmp0>0:
            tmp1 = tmp0//world_size + np.array([1]*(tmp0%world_size) + [0]*(world_size-(tmp0%world_size)))
            tmp2 = np.cumsum(np.concatenate([[0],tmp1]))
            tmp3 = index_list[-tmp0:]
            batch_index.append(tmp3[tmp2[rank]:tmp2[rank+1]])
        assert sum(len(x) for x in batch_index)==len(index_list[rank::world_size])
        # average last two batch two avoid len(batch)==0
        tmp0 = batch_index.pop(-1)
        tmp1 = batch_index.pop(-1)
        tmp0 = tmp1 + tmp0
        tmp1 = hf_ceil(len(tmp0), 2)
        batch_index.append(tmp0[:tmp1])
        batch_index.append(tmp0[tmp1:])
    else:
        num_batch = hf_ceil(len(index_list), batch_size)
        batch_index = [index_list[(x*batch_size):((x+1)*batch_size)] for x in range(num_batch)]
    assert len(batch_index)==num_batch
    assert all(len(x)>0 for x in batch_index)
    return batch_index


def my_create_validation_set(valdir, batch_size, target_size, rect_val, index_height_width, world_size=1, rank=0):
    if rect_val:
        sorted_idxar = sorted([(i, h/w) for i,h,w in index_height_width], key=lambda x:x[1])
        idx_sorted = [x[0] for x in sorted_idxar]
        batch_index = my_split_valdataset(idx_sorted, batch_size, world_size, rank)
        tmp0 = dict(sorted_idxar)
        chunk_mean = [sum(tmp0[y] for y in x)/len(x) for x in batch_index]
        idx2ar = {z:y for x,y in zip(batch_index,chunk_mean) for z in x}
        val_sampler = MySimpleSampler(batch_index)
        tmp0 = [torchvision.transforms.Resize(int(target_size*1.14)), CropArTfm(idx2ar, target_size)]
        val_dataset = ValDataset(valdir, transform=tmp0)
    else:
        tmp0 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(target_size*1.14)),
                torchvision.transforms.CenterCrop(target_size),
        ])
        val_dataset = torchvision.datasets.ImageFolder(valdir, transform=tmp0)
        val_sampler = MySimpleSampler(my_split_valdataset(list(range(len(val_dataset))), batch_size, world_size, rank))
    return val_dataset, val_sampler
