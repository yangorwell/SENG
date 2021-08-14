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
import math
import torch
import random
import imagesize #pip install imagesize
import tempfile
import warnings
import numpy as np
import PIL.Image
import nvidia.dali
import nvidia.dali.plugin.pytorch
import concurrent.futures


def generate_filepath_label_list(data_dir, extension=('jpg','jpeg','png'), tag_full_path=True):
    extension = set(extension)
    hf0 = lambda x: (x.lower().rsplit('.',1)[1] in extension)
    tmp0 = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,x))]
    if tag_full_path:
        folder_filepath_list = [(x,os.path.join(data_dir,x,y)) for x in tmp0
                    for y in os.listdir(os.path.join(data_dir, x)) if hf0(y)]
    else:
        folder_filepath_list = [(x,os.path.join(x,y)) for x in tmp0 for y in os.listdir(os.path.join(data_dir, x)) if hf0(y)]
    all_folder = {x for x,_ in folder_filepath_list}
    folder_to_id = {y:x for x,y in enumerate(sorted(all_folder))}
    filepath_label_list = sorted([(y,folder_to_id[x]) for x,y in folder_filepath_list])
    return folder_to_id, filepath_label_list

def _generate_shuffle_filelist(filepath_label_list, batch_size, world_size, rank, rng, tempdir):
    num_sample = len(filepath_label_list)
    num_batch = math.ceil(num_sample/(world_size*batch_size))
    num_padding = num_batch*world_size*batch_size - num_sample
    tmp0 = np.arange(num_sample)
    ind0 = np.concatenate([tmp0, rng.choice(tmp0,size=num_padding)]) #replace=True
    rng.shuffle(ind0)
    file_label_list_i = [filepath_label_list[x] for x in ind0.reshape(world_size, -1)[rank]]
    file_list_path = os.path.join(tempdir, f'dali_rank{rank}_{random.randint(0, 1000)}.txt')
    with open(file_list_path, 'w', encoding='utf-8') as fid:
        for x,y in file_label_list_i:
            fid.write(f'{x} {y}\n')
    return file_list_path, num_batch


class PaddedShuffleDALIPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self, data_dir, file_list_path, batch_size, min_crop_size, crop, mean, std, num_threads,
                device='gpu', world_size=1, rank=0, local_rank=0, seed=-1):
        assert device in {'cpu','gpu'}
        super(PaddedShuffleDALIPipeline, self).__init__(batch_size, num_threads, device_id=local_rank, seed=seed)
        self.source = nvidia.dali.ops.FileReader(file_root=data_dir, file_list=file_list_path, random_shuffle=False)
        tmp0 = {
            'device': 'mixed' if device=='gpu' else 'cpu',
            'output_type': nvidia.dali.types.RGB,
            'random_aspect_ratio': [0.8,1.25],
            'random_area': [min_crop_size,1],
            'device_memory_padding': 211025920 if device=='gpu' else 0,
            'host_memory_padding': 140544512 if device=='gpu' else 0,
            'memory_stats': False,
        }
        self.decode = nvidia.dali.ops.ImageDecoderRandomCrop(**tmp0)
        self.resize = nvidia.dali.ops.Resize(device=device, resize_x=crop, resize_y=crop)
        self.mirror_normalize = nvidia.dali.ops.CropMirrorNormalize(device=device,
                    dtype=nvidia.dali.types.FLOAT16, mean=mean, std=std)
        self.coin = nvidia.dali.ops.CoinFlip(probability=0.5)
        self.device = device

    def define_graph(self):
        image, label = self.source(name='source')
        image = self.decode(image)
        image = self.resize(image)
        image = self.mirror_normalize(image, mirror=self.coin())
        if self.device=='gpu':
            label = label.gpu()
        return image, label


# TODO iter start without ending the previous
class PaddedShuffleDALIDataLoader:
    def __init__(self, data_dir, batch_size, crop, min_crop_size, fp16=True, num_worker=6, seed=None, world_size=1, rank=0, local_rank=0):
        mean = (0.485*255, 0.456*255, 0.406*255)
        std = (0.229*255, 0.224*255, 0.225*255)
        # seed=-1 is okay for dali_pipeline now
        dali_pipe_kwargs = {'data_dir':data_dir, 'batch_size':batch_size,
                'min_crop_size':min_crop_size, 'crop':crop, 'mean':mean, 'std':std, 'seed':-1,
                'num_threads':num_worker, 'world_size':world_size, 'rank':rank, 'local_rank':local_rank}

        rng = np.random.RandomState(seed)
        tempdir = tempfile.TemporaryDirectory()
        folder_to_id, filepath_label_list = generate_filepath_label_list(data_dir, tag_full_path=False)
        file_list_path, num_batch = _generate_shuffle_filelist(filepath_label_list,
                    batch_size, world_size, rank, rng, tempdir.name)
        dali_pipe = PaddedShuffleDALIPipeline(file_list_path=file_list_path, **dali_pipe_kwargs)
        dali_iter = nvidia.dali.plugin.pytorch.DALIClassificationIterator(dali_pipe, reader_name='source')

        self.dali_pipe_kwargs = dali_pipe_kwargs
        self.rng = rng
        self.tempdir = tempdir
        self.folder_to_id = folder_to_id
        self.world_size = world_size
        self.rank = rank
        self.filepath_label_list = filepath_label_list
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.ind_step = 0
        self.dali_iter = dali_iter
        self.fp16 = fp16
        # self._new_dali_iter()
    def _new_dali_iter(self):
        del self.dali_iter
        file_list_path,_ = _generate_shuffle_filelist(self.filepath_label_list, self.batch_size,
                self.world_size, self.rank, self.rng, self.tempdir.name)
        dali_pipe = PaddedShuffleDALIPipeline(file_list_path=file_list_path, **self.dali_pipe_kwargs)
        self.dali_iter = nvidia.dali.plugin.pytorch.DALIClassificationIterator(dali_pipe, reader_name='source')
    def __len__(self):
        return self.num_batch
    def __iter__(self):
        if self.ind_step!=0:
            if self.rank==0:
                print('WARNING: PaddedShuffleDALIDataLoader not finished last iteration (generating a new one)')
            self._new_dali_iter()
            self.ind_step = 0
        return self
    def __next__(self):
        if self.ind_step==self.num_batch:
            self._new_dali_iter()
            self.ind_step = 0
            raise StopIteration
        tmp0 = next(self.dali_iter)[0]
        image = tmp0['data']
        if not self.fp16:
            image = image.float()
        label = tmp0['label'][:,0].to(torch.int64)
        self.ind_step = self.ind_step + 1
        return image,label


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
        tmp0 = batch_index.pop(-1) #-1
        tmp1 = batch_index.pop(-1) #-2
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


def resize_then_crop(HW0, target_size, scale=1.14):
    # if resize is smaller than crop_size, nvidia-dali will throw error
    HW0 = [np.asarray(x) for x in HW0]
    W_over_H = np.array([np.mean(x[:,1]/x[:,0]) for x in HW0])
    crop_HW = np.ones((len(HW0),2), dtype=np.int64)*target_size
    ind0 = W_over_H < 1
    crop_HW[ind0,0] = ((target_size/W_over_H[ind0]).astype(np.int64)//8) * 8
    ind1 = np.logical_not(ind0)
    crop_HW[ind1,1] = ((target_size*W_over_H[ind1]).astype(np.int64)//8) * 8

    resize_HW = []
    for hw,crop_hw in zip(HW0,crop_HW):
        tmp0 = np.min(hw/crop_hw, axis=1, keepdims=True)
        tmp1 = (np.ceil(np.ceil(scale*hw/tmp0)/8)*8).astype(np.int64)
        assert np.all(tmp1 >= crop_hw), (tmp1,crop_hw)
        resize_HW.append(tmp1)
    return resize_HW,crop_HW


class PaddedNoShuffleDALIPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self, data_dir, file_list_path, external_pipe, batch_size, mean, std, num_threads,
                device='gpu', world_size=1, rank=0, local_rank=0):
        super(PaddedNoShuffleDALIPipeline, self).__init__(batch_size, num_threads, device_id=local_rank, seed=233)
        self.source = nvidia.dali.ops.FileReader(file_root=data_dir, file_list=file_list_path, random_shuffle=False)
        self.external_pipe = nvidia.dali.ops.ExternalSource(source=external_pipe, num_outputs=5, cycle=True)

        tmp0 = {
            'device': 'mixed' if device=='gpu' else 'cpu',
            'output_type': nvidia.dali.types.RGB,
            'device_memory_padding': 211025920 if device=='gpu' else 0,
            'host_memory_padding': 140544512 if device=='gpu' else 0,
        }
        self.decode = nvidia.dali.ops.ImageDecoder(**tmp0)
        self.resize = nvidia.dali.ops.Resize(device=device)
        self.crop_normalize = nvidia.dali.ops.CropMirrorNormalize(device=device,
                    dtype=nvidia.dali.types.FLOAT16, mean=mean, std=std)
        self.device = device

    def define_graph(self):
        image, label = self.source(name='source')
        resize_h,resize_w,crop_h,crop_w,mask = self.external_pipe()

        image = self.decode(image)
        image = self.resize(image, resize_x=resize_w, resize_y=resize_h)
        image = self.crop_normalize(image, crop_h=crop_h, crop_w=crop_w)
        if self.device=='gpu':
            label = label.gpu()
            mask = mask.gpu()
        return image, label, mask


class MyExternalPipeline:
    def __init__(self, *args):
        assert len({len(x) for x in args}) == 1
        self.args = args
        self.num_batch = len(args[0])
        self.ind_step = None

    def __iter__(self):
        self.ind_step = 0
        return self

    def __next__(self):
        if self.ind_step==self.num_batch:
            raise StopIteration
        ret = tuple(x[self.ind_step] for x in self.args)
        self.ind_step = self.ind_step + 1
        return ret


# TODO iter start without ending the previous
class PaddedNoShuffleDALIDataLoader:
    def __init__(self, data_dir, batch_size, crop, rectangular=False, fp16=True, num_worker=6, world_size=1, rank=0, local_rank=0):
        folder_to_id, filepath_label_list = generate_filepath_label_list(data_dir, tag_full_path=False)
        mean = (0.485*255, 0.456*255, 0.406*255)
        std = (0.229*255, 0.224*255, 0.225*255)

        FLHW = [(x,y,*(imagesize.get(os.path.join(data_dir,x))[::-1])) for x,y in filepath_label_list]
        if rectangular:
            FLHW = sorted(FLHW, key=lambda x: x[3]/x[2])
            FLHW = my_split_valdataset(FLHW, batch_size, world_size, rank)
            HW0 = [np.array([(y[2],y[3]) for y in x]) for x in FLHW]
            resize_HW,crop_HW = resize_then_crop(HW0, crop)
            FLHW = [[(y0[0],y0[1],y1,y2,x2,x3) for y0,(y1,y2) in zip(x0,x1)] for x0,x1,(x2,x3) in zip(FLHW,resize_HW,crop_HW)]
        else:
            HW0 = np.array([x[2:] for x in FLHW])
            resize_HW = np.ones((len(HW0),2), dtype=np.int64) * int(1.14*crop)
            ind0 = HW0[:,0] < HW0[:,1]
            if np.any(ind0):
                resize_HW[ind0,1] = (np.ceil(np.ceil(HW0[ind0,1]/HW0[ind0,0] * resize_HW[ind0,0])/8)*8).astype(np.int64)
            ind1 = np.logical_not(ind0)
            if np.any(ind1):
                resize_HW[ind1,0] = (np.ceil(np.ceil(HW0[ind1,0]/HW0[ind1,1] * resize_HW[ind1,1])/8)*8).astype(np.int64)
            crop_HW = np.ones((len(HW0),2), dtype=np.int64) * crop
            FLHW = [(*x[:2],*y,*z) for x,y,z in zip(FLHW,resize_HW,crop_HW)]
            FLHW = my_split_valdataset(FLHW, batch_size, world_size, rank)
        FLHW = [[(*y,1) for y in x] for x in FLHW]
        if len(FLHW[-1]) < batch_size:
            tmp0 = FLHW[-1]
            tmp1 = (*tmp0[-1][:-1], 0)
            FLHW[-1] = tmp0 + [tmp1 for _ in range(batch_size-len(tmp0))]
        if len(FLHW[-2]) < batch_size:
            tmp0 = FLHW[-2]
            tmp1 = (*tmp0[-1][:-1], 0)
            FLHW[-2] = tmp0 + [tmp1 for _ in range(batch_size-len(tmp0))]
        tempdir = tempfile.TemporaryDirectory()
        file_list_path = os.path.join(tempdir.name, f'dali_rank{rank}_{random.randint(0, 1000)}.txt')
        with open(file_list_path, 'w', encoding='utf-8') as fid:
            for x,y in [y[:2] for x in FLHW for y in x]:
                fid.write(f'{x} {y}\n')

        tmp0 = np.array([[y[2:] for y in x] for x in FLHW]).transpose(2,0,1)[:,:,:,np.newaxis].copy()
        external_pipe = MyExternalPipeline(*(tmp0[:4].astype(np.float32)), tmp0[4])
        dali_pipe_kwargs = {'data_dir':data_dir, 'file_list_path':file_list_path, 'external_pipe':external_pipe,
                'batch_size':batch_size, 'mean':mean, 'std':std, 'num_threads':num_worker,
                'world_size':world_size, 'rank':rank, 'local_rank':local_rank}
        # dali_pipe = PaddedNoShuffleDALIPipeline(**dali_pipe_kwargs)
        # dali_iter = nvidia.dali.plugin.pytorch.DALIGenericIterator(dali_pipe, output_map=['data','label','mask'],
        #         dynamic_shape=rectangular, reader_name='source')

        self.tempdir = tempdir
        self.folder_to_id = folder_to_id
        self.external_pipe = external_pipe
        self.num_batch = external_pipe.num_batch
        self.rectangular = rectangular
        self.dali_pipe_kwargs = dali_pipe_kwargs
        self.ind_step = 0
        self.rank = rank
        self.dali_iter = None
        self.fp16 = fp16
        self._new_dali_iter()
    def _new_dali_iter(self):
        del self.dali_iter
        self.external_pipe.ind_step = 0
        dali_pipe = PaddedNoShuffleDALIPipeline(**self.dali_pipe_kwargs)
        self.dali_iter = nvidia.dali.plugin.pytorch.DALIGenericIterator(dali_pipe, output_map=['data','label','mask'],
                dynamic_shape=self.rectangular, reader_name='source')
    def __len__(self):
        return self.external_pipe.num_batch
    def __iter__(self):
        if self.ind_step!=0:
            if self.rank==0:
                print('WARNING: PaddedNoShuffleDALIDataLoader not finished last iteration')
            self.ind_step = 0
            self._new_dali_iter()
        return self
    def __next__(self):
        try:
            tmp0 = next(self.dali_iter)[0]
        except StopIteration:
            self.dali_iter.reset()
            self.ind_step = 0
            raise StopIteration
        ind0 = tmp0['mask'].sum()
        image = tmp0['data'][:ind0]
        if not self.fp16:
            image = image.float()
        label = tmp0['label'][:ind0,0].to(torch.int64)
        self.ind_step = self.ind_step + 1
        return image,label
