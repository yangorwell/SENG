# Pytorch:ImageNet-SENG Example

## Introduction

The Sketchy Empirical Natural Gradient Methods (**SENG**) is an algorithm for solving large-scale deep learning problems. It utilizes efficient sketching methods to reduce the computational and memory complexity of the Empirical Natural Gradient method.

Currently, **SENG** supports the convolution layer, batch normalization layer and fully-connected layer. 

This repository provides an **Pytorch** implementation of **SENG** about ImageNet Training.

## Requirements
* GPU with at least 8GB of memory.
* Python (>= 3.6)
* [pytorch (>= 1.5.0)](http://pytorch.org/)
* torchvision
* Scipy
* [nvidia-dali](https://developer.nvidia.com/DALI)
* tensorboardX
* imagesize


## Usage

- First install all the requirements and download the source code. 

- An example for training **ResNet-50** on **ImageNet2012** with multiple GPUs on one node:
  ```python
  python -m torch.distributed.launch --master_port 12115 --nproc_per_node=4 main_seng_bs.py --fp16 --lr 0.145  --lr_exponent 6 --damping 0.17  --logdir /path/to/logdir --epoch_end 60  --batch_size 64  --curvature_update_freq 1000 --fim_col_sample_size 128 -j 8 --datadir /path/to/dataset/ImageNet2012/
  
  ```

- To use **SENG** on other models, please refer to the `main_seng_bs.py` or the following codes as an example:

  ```python
  optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)
  preconditioner = SENG(net, 0.8, update_freq=50)

  for inputs, targets in trainloader:
     output = net(inputs)
     loss = criterion(outputs, targets)
     optimizer.zero_grad()
     loss.backward()

     preconditioner.step()
     optimizer.step()
  ```

- For more information, running `python main_seng_bs.py --help` outputs the following documentation:

  ```
  usage: main_seng_bs.py [-h] [--datadir DATADIR] [--logdir LOGDIR] [--lr LR] [--lr_warmup LR_WARMUP]
                         [--batch_size BATCH_SIZE] [--epoch EPOCH] [--epoch_end EPOCH_END]
                         [--lr_exponent LR_EXPONENT] [--momentum MOMENTUM]
                         [--weight_decay WEIGHT_DECAY] [--lr_decay_rate LR_DECAY_RATE]
                         [--warmup_epoch WARMUP_EPOCH] [--label_smoothing LABEL_SMOOTHING]
                         [--damping DAMPING] [--curvature_update_freq CURVATURE_UPDATE_FREQ]
                         [--fim_col_sample_size FIM_COL_SAMPLE_SIZE]
                         [--im_size_threshold IM_SIZE_THRESHOLD] [-j N] [--mixup MIXUP]
                         [--local_rank LOCAL_RANK] [--short-epoch] [--print_freq N] [--fp16]
                         [--use-dali]
  
  SENG optimizer
  
  optional arguments:
    -h, --help            show this help message and exit
    --datadir DATADIR     Place where data are stored
    --logdir LOGDIR       where logs go
    --lr LR               learning rate
    --lr_warmup LR_WARMUP
                          learning rate warmup
    --batch_size BATCH_SIZE, -b BATCH_SIZE
                          batch size per gpu
    --epoch EPOCH         epoch
    --epoch_end EPOCH_END
    --lr_exponent LR_EXPONENT
    --momentum MOMENTUM   momentum
    --weight_decay WEIGHT_DECAY
                          weight_decay
    --lr_decay_rate LR_DECAY_RATE
                          the rate damping is decayed
    --warmup_epoch WARMUP_EPOCH
                          first k epoch to gradually increase learning rate
    --label_smoothing LABEL_SMOOTHING
                          label smoothing parameter
    --damping DAMPING     damping
    --curvature_update_freq CURVATURE_UPDATE_FREQ
                          The frequency to update inverse fisher matrix
    --fim_col_sample_size FIM_COL_SAMPLE_SIZE
                          subsample count of col
    --im_size_threshold IM_SIZE_THRESHOLD
                          only approximate over this size
    -j N, --workers N     number of data loading workers
    --mixup MIXUP         mixup interpolation coefficient
    --local_rank LOCAL_RANK
                          provided by torch.distributed.launch
    --short-epoch         make epochs short (for debugging)
    --print_freq N        log/print every this many steps
    --fp16                Run model fp16 mode
    --use-dali            use nvidia.dali gpu mode to do data loading work
  ```

## The Authors
If you have any bug reports or comments, please feel free to email one of the authors:

* Minghan Yang, yangminghan at pku.edu.cn
* Dong Xu, taroxd at pku.edu.cn
* Zaiwen Wen, wenzw at pku.edu.cn

## License

This package is released under [GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html).
