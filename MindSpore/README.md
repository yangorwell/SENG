
# MindSpore-ResNet-50-SENG Example

This version of code is written in Mindspore v1.2.  Codes are available in https://gitee.com/taroxd/mindspore/tree/seng/model_zoo/official/cv/resnet50_seng




## Description

This is an example of training ResNet-50 V1.5 with ImageNet2012 dataset by second-order optimizer SENG. SENG is a novel approximate seond-order optimization method. With fewer iterations, SENG can finish ResNet-50 V1.5 training in 273 minutes within 41 epochs to top-1 accuracy of 75.9% using 4 Tesla V100 GPUs, which is much faster than SGD with Momentum.

## Model Architecture

The overall network architecture of ResNet-50 is show below:[link](https://arxiv.org/pdf/1512.03385.pdf)

## Dataset

Dataset used: ImageNet2012

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py

- Download the dataset ImageNet2012

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should include train and eval dataset as follows:

```shell
    ├── ilsvrc                  # train dataset
    └── ilsvrc_eval             # infer dataset
```

## Environment Requirements

- Hardware（GPU）
    - Prepare hardware environment with GPU processors.

- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

Running on GPU (cifar10, untuned, single GPU)

```bash
# run training example
python train.py --data_path "/path/to/cifar-10/cifar-10-batches-bin" --ckpt_save_path "/userhome/mindspore_xd_cifar10/ms-seng/ckpt_debug" --epoch_size 5 --lr_decay_mode linear --lr_init 0.01 --lr_max 0.01 --lr_end 0.0001 --batch_size 32 --damping_init 0.5 --damping_decay 0.8 --momentum 0.9 --loss_scale 1024 --weight_decay 5e-4 --frequency 71 --im_size_threshold 1000000 --save_ckpt 1  

# evaluate on training set
python eval.py --ckpt_path "/path/to/ckpt" --data_path "/path/to/cifar-10/cifar-10-batches-bin" --dataset cifar10 --is_train 1

# run evaluation example
python eval.py --ckpt_path "/path/to/ckpt" --data_path "/path/to/cifar-10/cifar-10-verify-bin" --dataset cifar10 --is_train 0
```

Running on GPU (ImageNet2012, distributed on 4 GPUs)

```bash
# run distributed training example
mpirun --allow-run-as-root -n 4 python train.py --data_path "/path/to/ImageNet2012/train" --epoch_size 45 --ckpt_save_path "/path/to/ckpt" --device_num 4 --label_smoothing 0.1 --dataset imagenet2012 --lr_init 0.18 --lr_max 48 --lr_end 4.5 --decay_epochs 40 --lr_decay_mode exp --batch_size 64 --momentum 0.9 --warmup_epoch 5 --loss_scale 128 --weight_decay 1e-4 --frequency 1668 --damping_init 0.2 --damping_decay 0.8 --im_size_threshold 700000 --col_sample_size 128 --save_ckpt 1

# run evaluation example
python eval.py --ckpt_path "/path/to/ckpt" --data_path "/path/to/ImageNet2012/val" --label_smoothing 0.1 --dataset imagenet2012 --is_train 0
```

## Contact 

We hope that the package is useful for your application. If you have any bug reports or comments, please feel free to email one of the toolbox authors:

- Minghan Yang, yangminghan at pku.edu.cn
- Dong Xu, taroxd at pku.edu.cn
- Zaiwen Wen, wenzw at pku.edu.cn

## Reference

Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu, Sketchy Empirical Natural Gradient Methods for Deep Learning,  https://arxiv.org/abs/2006.05924