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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
import sys
import os
import time
import warnings
import math

import numpy as np

from models.resnet_ex import resnet50,resnet18
from models.vgg_ex import vgg16,vgg11,vgg16_bn
from utils.dist_utils import rank0_print
from label_smoothing_loss import LabelSmoothingLoss

from seng import SENG

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--datadir', default='/datasets', help='Place where data are stored')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-decay-epoch', default=30, type=int, help='learning rate decay at n epoches')
parser.add_argument('--lr-decay-rate', default=0.1, type=float, help='how much learning rate decays')
parser.add_argument('--lr-scheme','--lr-schedule',default='staircase', type=str, help='how much learning rate decays')
parser.add_argument('--batch-size', '-b', default=256, type=int, help='batch size across all nodes')
parser.add_argument('--epoch', default=100, type=int, help='epoch')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--damping', default=0.05, type=float, help='initial damping')
parser.add_argument('--curvature-update-freq', default=200, type=int,
                    help='The frequency to update inverse fisher matrix [default 50]')
parser.add_argument('--fim-subsample', type=int, help='subsample count of GPU')
parser.add_argument('--fim-col-sample-size', type=int, default=256, help='subsample count of col')
parser.add_argument('--im-size-threshold', type=int, default=700000, help='only approximate over this size')
parser.add_argument('--label-smoothing', default=0.0, type=float, help='label smoothing parameter')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--trainset', default='cifar10',
                    choices=['cifar10', 'cifar100','imagenette','svhn'],
                    help='training dataset')
parser.add_argument('--arch', default='vgg16_bn',
                    choices=['vgg16', 'vgg16_bn', 'resnet50', 'resnet18'],
                    help='model architecture')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--verbose', action='store_true',
                    help='print process when training')


args = parser.parse_args()

def lr_schedule(epoch, lr0):
    lr = lr0
    if args.lr_scheme == 'staircase':
        lr = lr * (args.lr_decay_rate**(epoch // args.lr_decay_epoch))
    elif args.lr_scheme == 'cosine':
    # cosine
        epoch_tune = args.lr_decay_epoch
        if epoch < epoch_tune:
            lr = 0.001 + 0.5 * (lr - 0.001) * (1 + math.cos(epoch / epoch_tune * math.pi))
        else:
            lr = 0.0005
    else:
        lr = lr * (1.0 - epoch/args.lr_decay_epoch)**args.lr_decay_rate
    return lr

def adjust_learning_rate(optimizer, epoch, args):
    lr = lr_schedule(epoch, args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_damping(preconditioner, epoch, args):
    #damping = args.damping * (args.lr_decay_rate**((epoch // args.lr_decay_epoch) / 5))
    damping = args.damping ## constant damping strategy. One can tune the damping strategy.  
    preconditioner.damping = damping


def main():
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)



    rank0_print("==> Running with {0}".format(sys.argv))

    cudnn.benchmark = True

    rank0_print('==> Building model..')

    if args.trainset == 'cifar10' or args.trainset=='imagenette':
        datamean = (0.4914, 0.4822, 0.4465)
        datastd = (0.2470, 0.2435, 0.2616)
        num_classes = 10
    elif args.trainset == 'cifar100':
        datamean = (0.5071, 0.4867, 0.4408)
        datastd = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
    	num_classes = 10

    if args.arch == 'resnet50':
        net = resnet50(num_classes=num_classes)
    elif args.arch == 'resnet18':
        net = resnet18(num_classes=num_classes)
    elif args.arch == 'vgg16_bn':
        net = vgg16_bn(num_classes=num_classes)
    elif args.arch == 'vgg16':
        net = vgg16(num_classes=num_classes)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        else:
            net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)

    rank0_print('==> Preparing data..')

    if args.trainset == 'imagenette' :
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.datadir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(traindir,transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        testset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.trainset == 'svhn':
        transform_train=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        transform_test=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(datamean, datastd),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(datamean, datastd),
        ])

    if args.trainset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=False, transform=transform_test)
    elif args.trainset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=args.datadir, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.datadir, train=False, download=False, transform=transform_test)
    elif args.trainset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=args.datadir, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=args.datadir, split='test', download=True, transform=transform_test)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(args.label_smoothing).cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    preconditioner = SENG(net, args.damping, update_freq=args.curvature_update_freq, verbose=args.verbose, subsample=args.fim_subsample, im_size_threshold=args.im_size_threshold, col_sample_size=args.fim_col_sample_size)

    pending_batch = None

    # Training
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            num_iter = preconditioner.iteration_counter
            epoch_for_adjust = epoch + (batch_idx + 1) / len(trainloader)
            adjust_learning_rate(optimizer, epoch_for_adjust, args)
            adjust_damping(preconditioner, epoch_for_adjust, args)
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            loss.backward()

            preconditioner.step()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            this_batch_time = time.time() - epoch_start_time

            if args.verbose:
                if num_iter % 50 == 0:
                    rank0_print('%3d-%4d   %2.1e  %2.1e  %2.1e  %2.1e   %2.1e  %3.1f%%' %
                    (epoch, num_iter, loss.item(), preconditioner.state['normg'], preconditioner.state['normd'],  preconditioner.state['adg'], preconditioner.damping, correct / total * 100))
        return train_loss / len(trainloader), correct / total



    def validate(epoch):
        # global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if args.gpu is not None:
                    inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return test_loss / len(testloader), correct / total

    total_time = 0

    for epoch in range(args.epoch):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = train(epoch)
        train_time = time.time() - start_time
        total_time += train_time
        test_loss, test_acc = validate(epoch)

        if epoch == 0:
            mstats = torch.cuda.memory_stats()
            rank0_print('Memory peak: %d Bytes' % mstats['active_bytes.all.peak'])
            rank0_print('Epoch  testloss  testacc  trainloss  trainacc  time')



        rank0_print("  %2d   %6.4f     %6.3f  %6.4f    %6.3f  %.2f" % (
            epoch + 1, test_loss,
            test_acc * 100, train_loss,
            train_acc * 100, total_time), flush=True)

if __name__ == '__main__':
    main()
