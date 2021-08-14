# Copyright (c) 2021 Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu
# All rights reserved.
import os
import time
import logging
import torch
from tensorboardX import SummaryWriter


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass
        return no_op


class TensorboardLogger:

    def __init__(self, output_dir):
        self.scaler_json_filepath = os.path.join(output_dir, 'scalars.json')
        self.writer = SummaryWriter(output_dir)
        self.current_step = 0

    def log(self, tag, val):
        self.writer.add_scalar(tag, val, self.current_step)

    def log_epoch(self, epoch, batch_size, image_size):
        # at the beginning of each epoch
        self.log('epoch', epoch)
        self.log('size/batch_size', batch_size)
        self.log('size/image_size', image_size)

    def log_train(self, lr, batch_size, loss, acc1, acc5):
        # at the end of train step
        self.log('size/lr', lr)
        self.log('size/batch_size', batch_size)
        self.log('loss/train_loss', loss)
        self.log("loss/train_1", acc1)
        self.log("loss/train_5", acc5)

    def log_eval(self, acc1, acc5, time):
        # at the end of validation epoch
        self.log('loss/val_acc1', acc1)
        self.log('loss/val_acc5', acc5)
        self.log('time/val', time)

    def log_time_memory(self, batch_size, data_time, fwd_time):
        self.log('time/data', data_time)
        self.log('time/fwd', fwd_time)
        self.log('time/image_per_sec', batch_size/(data_time+fwd_time))
        self.log('memory/allocated_gb', torch.cuda.memory_allocated()/1e9)
        self.log('memory/max_allocated_gb', torch.cuda.max_memory_allocated()/1e9)
        self.log('memory/cached_gb', torch.cuda.memory_reserved()/1e9)
        self.log('memory/max_cached_gb', torch.cuda.max_memory_reserved()/1e9)

    def log_network(self, recv_gbit, transmit_gbit):
        self.log('net/recv_gbit', recv_gbit)
        self.log('net/transmit_gbit', transmit_gbit)

    def step(self, batch_size):
        # at the end of step
        self.current_step += batch_size

    def close(self):
        self.writer.export_scalars_to_json(self.scaler_json_filepath)
        self.writer.close()


class FileLogger:

    def __init__(self, log_to_file, output_dir, name='imagenet_training'):
        # is_master: log_to_console, log_to_file
        # is_local_rank0: log_to_console
        # otherwise: NoOp
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if log_to_file:
            vlog = logging.FileHandler(output_dir+'/info.log')
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            logger.addHandler(vlog)

            tmp0 = logging.FileHandler(output_dir+'/warn.log')
            tmp0.setLevel(logging.WARN)
            tmp0.setFormatter(formatter)
            logger.addHandler(tmp0)

            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir+'/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        self.logger = logger

    def debug(self, *args):
        self.logger.debug(*args)

    def warn(self, *args):
        self.logger.warn(*args)

    def info(self, *args):
        self.logger.info(*args)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, avg_mom=0.5):
        self.avg_mom = avg_mom
        self.val = 0
        self.avg = 0 # running average of whole epoch
        self.smooth_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.smooth_avg = val if self.count==0 else (self.avg*self.avg_mom + val*(1-self.avg_mom))
        self.avg = self.sum / self.count


class TimeMeter:
    def __init__(self):
        self.fwd_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        t0 = time.time()
        self.data_time.update(t0-self.start)
        self.start = t0

    def batch_end(self):
        t0 = time.time()
        self.fwd_time.update(t0 - self.start)
        self.start = t0


def network_bytes():
    with open('/proc/net/dev', 'r', encoding='ascii') as fid:
        lines = [x.strip() for x in fid]
    tmp0 = [x.split() for x in lines[2:] if not x.startswith('lo')] #ignore loopback interface
    received_bytes = sum(int(x[1]) for x in tmp0)
    transmitted_bytes = sum(int(x[9]) for x in tmp0)
    return received_bytes, transmitted_bytes


class NetworkMeter:
    def __init__(self):
        self.recv_meter = AverageMeter()
        self.transmit_meter = AverageMeter()
        self.last_recv_bytes, self.last_transmit_bytes = network_bytes()
        self.last_log_time = time.time()

    def update_bandwidth(self):
        time_delta = time.time() - self.last_log_time
        recv_bytes, transmit_bytes = network_bytes()

        recv_delta = recv_bytes - self.last_recv_bytes
        transmit_delta = transmit_bytes - self.last_transmit_bytes

        # turn into Gbps
        recv_gbit = 8*recv_delta/time_delta/1e9
        transmit_gbit = 8*transmit_delta/time_delta/1e9
        self.recv_meter.update(recv_gbit)
        self.transmit_meter.update(transmit_gbit)

        self.last_log_time = time.time()
        self.last_recv_bytes = recv_bytes
        self.last_transmit_bytes = transmit_bytes
        return recv_gbit, transmit_gbit
