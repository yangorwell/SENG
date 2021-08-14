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
import math
import warnings
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

warnings.filterwarnings("ignore", ("Please set `reader_name` and don't set last_batch_padded and size manually "
                " whenever possible. This may lead, in some situations, to miss some "
                " samples or return duplicated ones. Check the Sharding section of the "
                "documentation for more details."), Warning)


class HybridTrainPipe(Pipeline):
    """
    DALI Train Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip

    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    min_crop_size (float, optional, default = 0.08) - Minimum random crop size
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 mean, std, rank=0, world_size=1, dali_cpu=False, shuffle=True, fp16=False,
                 min_crop_size=0.08, ext_data_source=False):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)

        self.world_size = world_size
        self.ext_data_source = ext_data_source

        # Enabling read_ahead slowed down processing ~40%
        self.input = ops.FileReader(file_root=data_dir, shard_id=rank, num_shards=world_size, random_shuffle=shuffle)
        # Let user decide which pipeline works best with the chosen model
        if dali_cpu:
            decode_device = "cpu"
            self.dali_device = "cpu"
            self.flip = ops.Flip(device=self.dali_device)
        else:
            decode_device = "mixed"
            self.dali_device = "gpu"

            output_dtype = types.FLOAT
            if self.dali_device == "gpu" and fp16:
                output_dtype = types.FLOAT16

            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=output_dtype,
                                               output_layout=types.NCHW,
                                               crop=(crop, crop),
                                            #    image_type=types.RGB,
                                               mean=mean,
                                               std=std,)

        # To be able to handle all images from full-sized ImageNet, this padding sets the size of the internal
        # nvJPEG buffers without additional reallocations
        device_memory_padding = 211025920 if decode_device == 'mixed' else 0
        host_memory_padding = 140544512 if decode_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decode_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[min_crop_size, 1.0],
                                                 num_attempts=100)

        # Resize as desired.  To match torchvision data loader, use triangular interpolation.
        self.res = ops.Resize(device=self.dali_device, resize_x=crop, resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)

        self.coin = ops.CoinFlip(probability=0.5)
        # print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        # Combined decode & random crop
        images = self.decode(self.jpegs)

        # Resize as desired
        images = self.res(images)

        if self.dali_device == "gpu":
            output = self.cmn(images, mirror=rng)
        else:
            # CPU backend uses torch to apply mean & std
            output = self.flip(images, horizontal=rng)

        self.labels = self.labels.gpu()
        return [output, self.labels]

    @property
    def num_samples(self):
        return int(self.epoch_size("Reader") / self.world_size)

class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, **kwargs):
        # set file_last_batch and last_batch_padded to keep the same as pytorch
        self._dali_iterator = DALIClassificationIterator(pipelines=pipelines, size=pipelines.num_samples,
                                        fill_last_batch=False, last_batch_padded=True)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            # print('Resetting DALI loader')
            self._dali_iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()

        return input, target
