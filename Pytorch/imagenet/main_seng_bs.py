# Copyright (c) 2021 Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu
# All rights reserved.
import os
import datetime
import numpy as np
import torch
import torchvision

torch.backends.cudnn.benchmark = True

from resnet_seng import resnet50
from seng_dist import SENG
from utils import MPIEnv, get_parser_seng, check_phase, save_file_for_reproduce
from utils import LabelSmoothingLoss, LRScheduler, my_topk, sum_tensor
from logging_utils import TensorboardLogger, FileLogger, TimeMeter, AverageMeter, NoOp
import data_manager

SAVE_LIST = [os.path.abspath(__file__), 'seng_dist.py', 'resnet_seng.py',
        'logging_utils.py', 'main_seng_bs.py','utils.py', 'data_manager.py', 'dali_pipe.py']


def split_weights(net):
    decay = []
    no_decay = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
    assert len(list(net.parameters())) == len(decay) + len(no_decay)
    ret = [{'params':decay}, {'params':no_decay, 'weght_decay':0}]
    return ret

def train(train_loader):
    timer = TimeMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    net.train()
    for ind_batch, (data_i, label_i) in enumerate(train_loader, start=1):
        if args.short_epoch and (ind_batch>(11*args.print_freq+1)):
            break
        timer.batch_start()
        scheduler.update_lr(ind_epoch, ind_batch, len(train_loader),args.lr_exponent,args.epoch_end,args.lr_decay_rate)
        if ind_batch==1:
            file_logger.warn(f'Changing LR from {scheduler.lr_epoch_start} to {scheduler.lr_epoch_end}')
        tmp0 = ind_epoch + ind_batch / len(train_loader)
        preconditioner.damping = args.damping * (args.lr_decay_rate**(tmp0 / 10))

        if loss_scaler is not None:
            with torch.cuda.amp.autocast():
                predict_i = net(data_i) #float16
                loss = criterion(predict_i, label_i) #float32
            optimizer.zero_grad()
            loss_scaler.scale(loss).backward()
            preconditioner.step()
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            predict_i = net(data_i)
            loss = criterion(predict_i, label_i)
            optimizer.zero_grad()
            loss.backward()
            preconditioner.step()
            optimizer.step()

        timer.batch_end()
        acc1, acc5 = my_topk(predict_i.detach(), label_i, topk=(1,5))
        loss = loss.item()
        batch_size = data_i.shape[0]
        if args.mpi.is_distributed:
            metrics = torch.tensor([batch_size, loss, acc1, acc5]).float().cuda()
            batch_size, loss, acc1, acc5 = sum_tensor(metrics).cpu().numpy()
            acc1 = acc1.item()
            acc5 = acc5.item()
            loss = loss / args.mpi.world_size
        loss_meter.update(loss, batch_size)
        acc1_meter.update(100*acc1/batch_size, batch_size)
        acc5_meter.update(100*acc5/batch_size, batch_size)

        if (ind_batch%args.print_freq==0) or (ind_batch==len(train_loader)):
            tb_logger.log_time_memory(batch_size, timer.data_time.val, timer.fwd_time.val)
            tb_logger.log_train(optimizer.param_groups[0]['lr'], batch_size, loss, acc1, acc5)
            tmp0 = (f'[Train][{ind_epoch}][{ind_batch}/{len(train_loader)}]\t'
                    f'Data {timer.data_time.val:.3f} ({timer.data_time.avg:.3f})\t'
                    f'Forward {timer.fwd_time.val:.3f} ({timer.fwd_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
            file_logger.info(tmp0)
        tb_logger.step(batch_size)

    file_logger.warn(f'[epoch={ind_epoch}] train: time={timer.fwd_time.sum+timer.data_time.sum:.0f} seconds, '
                    f'acc@1={acc1_meter.avg:.3f}%, acc@5={acc5_meter.avg:.3f}%\n')


def validate(val_loader):
    timer = TimeMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    net.eval()
    with torch.no_grad():
        for ind_batch, (data_i, label_i) in enumerate(val_loader, start=1):
            if args.short_epoch and (ind_batch>(3*args.print_freq+1)):
                break
            timer.batch_start()

            if args.fp16:
                with torch.cuda.amp.autocast():
                    predict_i = net(data_i)
                    loss = criterion(predict_i, label_i)
            else:
                predict_i = net(data_i)
                loss = criterion(predict_i, label_i)

            timer.batch_end()
            acc1, acc5 = my_topk(predict_i.detach(), label_i, topk=(1,5))
            loss = loss.item()
            batch_size = data_i.shape[0]
            if args.mpi.is_distributed:
                tmp0 = torch.tensor([batch_size, loss, acc1, acc5]).float().cuda()
                batch_size, loss, acc1, acc5 = sum_tensor(tmp0).cpu().numpy().tolist()
                loss = loss / args.mpi.world_size
            timer.batch_end()
            loss_meter.update(loss, batch_size)
            acc1_meter.update(100*acc1/batch_size, batch_size)
            acc5_meter.update(100*acc5/batch_size, batch_size)

            if (ind_batch%args.print_freq==0) or (ind_batch==len(val_loader)):
                tb_logger.log_time_memory(batch_size, timer.data_time.val, timer.fwd_time.val)
                tmp0 = (f'[Test][{ind_epoch}][{ind_batch}/{len(val_loader)}]\t'
                        f'Data {timer.data_time.val:.3f} ({timer.data_time.avg:.3f})\t'
                        f'Forward {timer.fwd_time.val:.3f} ({timer.fwd_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
                file_logger.info(tmp0)

    tmp0 = timer.fwd_time.sum+timer.data_time.sum
    tb_logger.log_eval(acc1_meter.avg, acc1_meter.avg, tmp0)
    file_logger.warn(f'[epoch={ind_epoch}] validation: time={tmp0:.0f} seconds, '
                    f'acc@1={acc1_meter.avg:.3f}%, acc@5={acc5_meter.avg:.3f}%\n')
    return acc1_meter.avg, acc5_meter.avg


if __name__ == '__main__':
    args = get_parser_seng().parse_args()
    lr = 0.1
    lr_phase = [
        {'ep':(0,args.warmup_epoch), 'lr':(args.lr_warmup,args.lr), 'type':'linear'}, # lr warmup is better with --init-bn0
        {'ep':(args.warmup_epoch,args.epoch), 'lr':(args.lr,args.lr/10), 'type':'exp'}, # trying one cycle
    ]
    dl_phase = [
        {'ep':0, 'sz':224, 'bs':args.batch_size, 'val_bs':256, 'min_scale':0.087},
    ]
    args.lr_phase, args.dl_phase = check_phase(lr_phase, dl_phase)
    args.mpi = MPIEnv()
    if args.mpi.is_master:
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        tb_logger = TensorboardLogger(output_dir=args.logdir)
        file_logger = FileLogger(log_to_file=True, output_dir=args.logdir)
        save_file_for_reproduce(SAVE_LIST, args.logdir)
    elif args.mpi.is_local_rank0:
        tb_logger = NoOp()
        file_logger = FileLogger(log_to_file=False, output_dir=args.logdir)
    else:
        tb_logger = NoOp()
        file_logger = NoOp()
    torch.cuda.set_device(args.mpi.local_rank)
    file_logger.warn(args)

    net = resnet50(num_classes=1000, zero_init_residual=True).cuda()
    if args.mpi.is_distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.mpi.local_rank])

    if args.fp16:
        loss_scaler = torch.cuda.amp.GradScaler(1024) #positive power of 2 values can improve fp16 convergence
    else:
        loss_scaler = None

    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(split_weights(net), lr=args.lr_phase[0]['lr'][0],
            momentum=args.momentum, weight_decay=args.weight_decay)
    preconditioner = SENG(net, args.damping, update_freq=args.curvature_update_freq,
            im_size_threshold=args.im_size_threshold, col_sample_size=args.fim_col_sample_size,
            world_size=args.mpi.world_size, loss_scaler=loss_scaler)
    dm = data_manager.DataManager(phases=args.dl_phase, ilsvrc_root=args.datadir, workers=args.workers,
            use_dali=True, fp16=args.fp16, world_size=args.mpi.world_size, rank=args.mpi.rank,
            local_rank=args.mpi.local_rank)
    scheduler = LRScheduler(optimizer, args.lr_phase)

    start_time = datetime.datetime.now()
    for ind_epoch in range(max(x['ep'][1] for x in args.lr_phase)):
        dm.set_epoch(ind_epoch)
        file_logger.warn(f"[epoch={ind_epoch}] image size: {dm.current_phase['sz']}")
        file_logger.warn(f"[epoch={ind_epoch}] batch size: {dm.current_phase['bs']}")
        tb_logger.log_epoch(ind_epoch, dm.current_phase['bs'], dm.current_phase['sz'])

        train(dm.trn_dl)
        acc1, acc5 = validate(dm.val_dl)

        tmp0 = (datetime.datetime.now()-start_time).total_seconds()
        file_logger.warn(f'[epoch={ind_epoch}] summary: time={tmp0:.0f} seconds, acc@1={acc1:.3f}%, acc@5={acc5:.3f}%\n')

    tb_logger.close()
