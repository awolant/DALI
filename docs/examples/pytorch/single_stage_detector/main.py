import os
import sys
import time
from argparse import ArgumentParser
import math
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed


from src.model import model, Loss
from src.utils import dboxes300_coco, Encoder

from src.evaluate import evaluate
from src.train import train_loop, tencent_trick
from src.data import *

# Apex imports
try:
    from apex.parallel.LARC import LARC
    from apex import amp
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


class Logger:
    def __init__(self, batch_size, local_rank, print_freq=20):
        self.batch_size = batch_size
        self.local_rank = local_rank
        self.print_freq = print_freq

        self.processed_samples = 0
        self.epochs_times = []
        self.epochs_speeds = []


    def update_iter(self, epoch, iteration, loss):
        if self.local_rank != 0:
            return

        if iteration % self.print_freq == 0:
            print('Epoch: {:2d}, Iteraion: {}, Loss: {}'.format(epoch, iteration, loss))

        self.processed_samples = self.processed_samples + self.batch_size

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self):
        epoch_time = time.time() - self.epoch_start
        epoch_speed = self.processed_samples / epoch_time

        self.epochs_times.append(epoch_time)
        self.epochs_speeds.append(epoch_speed)
        self.processed_samples = 0

        if self.local_rank == 0:
            print('Epoch {:2d}, Time: {:4f} s, Speed: {:4f} img/sec, Average speed: {:4f}'
                .format(len(self.epochs_times), epoch_time, epoch_speed, self.average_speed()))

    def average_speed(self):
        return sum(self.epochs_speeds) / len(self.epochs_speeds)


def make_parser():
    parser = ArgumentParser(
        description="Train Single Shot MultiBox Detector on COCO")
    parser.add_argument(
        '--data', '-d', type=str, default='/coco', required=True,
        help='path to test and training data files')
    parser.add_argument(
        '--epochs', '-e', type=int, default=65,
        help='number of epochs for training')
    parser.add_argument(
        '--batch-size', '--bs', type=int, default=32,
        help='number of examples for each iteration')
    parser.add_argument(
        '--eval-batch-size', '--ebs', type=int, default=32,
        help='number of examples for each evaluation iteration')
    parser.add_argument(
        '--seed', '-s', type=int, default=0,
        help='manually set random seed for torch')
    parser.add_argument(
        '--evaluation', nargs='*', type=int,
        default=[3, 21, 31, 37, 42, 48, 53, 59, 64],
        help='epochs at which to evaluate')
    parser.add_argument(
        '--multistep', nargs='*', type=int, default=[43, 54],
        help='epochs at which to decay learning rate')
    parser.add_argument(
        '--target', type=float, default=None,
        help='target mAP to assert against at the end')

    # Hyperparameters
    parser.add_argument(
        '--learning-rate', '--lr', type=float, default=2.6e-3, help='learning rate')
    parser.add_argument(
        '--momentum', '-m', type=float, default=0.9,
        help='momentum argument for SGD optimizer')
    parser.add_argument(
        '--weight-decay', '--wd', type=float, default=0.0005,
        help='momentum argument for SGD optimizer')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument(
        '--backbone', type=str, default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--fp16-mode', type=str, default='amp', choices=['off', 'static', 'amp'],
        help='Half precission mode to use')

    # Distributed
    parser.add_argument('--local_rank', default=0, type=int,
        help='Used for multi-process training. Can either be manually set ' +
            'or automatically set by using \'python -m multiproc\'.')

    # Pipeline control
    parser.add_argument(
        '--data_pipeline', type=str, default='dali', choices=['dali', 'no_dali'],
        help='data preprocessing pipline to use')

    return parser


def train(args):
    if args.amp:
        amp_handle = amp.init(enabled=args.fp16)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    ssd300 = model(args)
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    iteration = 0
    loss_func = Loss(dboxes)

    loss_func.cuda()

    optimizer = torch.optim.SGD(
        tencent_trick(ssd300), 
        lr=args.learning_rate,
        momentum=args.momentum,
         weight_decay=args.weight_decay)

    scheduler = MultiStepLR(
        optimizer=optimizer, 
        milestones=args.multistep, 
        gamma=0.1)

    if args.fp16:
        if args.amp:
            optimizer = amp_handle.wrap_optimizer(optimizer)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.)

    val_dataloader, inv_map = get_val_dataloader(args)
    train_loader = get_train_loader(args, dboxes)

    acc = 0
    logger = Logger(args.batch_size, args.local_rank)
    
    for epoch in range(0, args.epochs):
        logger.start_epoch()
        scheduler.step()

        iteration = train_loop(
            ssd300, loss_func, epoch, optimizer, 
            train_loader, iteration, logger, args)

        logger.end_epoch()

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
            if args.local_rank == 0:
                print('Epoch {:2d}, Accuracy: {:4f} mAP'.format(epoch, acc))

        train_loader.reset()

    return acc, logger.average_speed()
        

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    torch.backends.cudnn.benchmark = True

    if args.fp16_mode != 'off':
        args.fp16 = True
        args.amp = (args.fp16_mode == 'amp')
    else:
        args.fp16 = False
        args.amp = False

    start_time = time.time()
    acc, avg_speed = train(args)
    training_time = time.time() - start_time

    if args.local_rank == 0:
        print("Training end: Average speed: {:3f} img/sec, Total time: {:3f} sec, Final accuracy: {:3f} mAP"
          .format(avg_speed, training_time, acc))

        if args.target is not None:
            if args.target > acc:
                print('Target mAP of {} not met. Possible regression'.format(args.target))
                sys.exit(1)
