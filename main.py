from __future__ import division
import argparse
import os
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import alexnet as modifiednet
import vgg16 as modified_vgg16net


# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc

cwd = os.getcwd()
sys.path.append(cwd + '/../')
import datasets as datasets
import datasets.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    help='model architecture (default: alexnet)')
parser.add_argument('--data', metavar='DATA_PATH',
                    default='./home/yuxiang/FiberTrack/YX_data/generated_ellipse/Lan/image/',
                    help='path to imagenet data (default: ./home/yuxiang/FiberTrack/YX_data/generated_ellipse/Lan/image/)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # need to modify when it is not
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=True, help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

use_gpu = torch.cuda.is_available()


def main():
    global args
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    # create model
    if args.arch == 'alexnet':
        model = modifiednet.main(args.arch)

    if args.arch == 'vgg16':
        model = modified_vgg16net.main(args.arch)

    if use_gpu:
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()  # loss for classification
    # smooth_l1 = nn.SmoothL1Loss()  # loss for regression
    # mse = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a pretrained model(checkpoint)
    if args.resume:
        # if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # load train data
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomSizedCrop(input_size),
        ]),
        'val': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomSizedCrop(input_size),
        ])
    }

    # need to define

    num_class = 3
    train_dataset = datasets.customData(img_path='',
                                        txt_path='/home/hongkai/PyTorch_Tutorial/pytorch_MultiTask-two-branches/data/train.txt',
                                        data_transforms=data_transforms,
                                        dataset='train')

    val_dataset = datasets.customData(img_path='',
                                      txt_path='/home/hongkai/PyTorch_Tutorial/pytorch_MultiTask-two-branches/data/val.txt',
                                      data_transforms=data_transforms,
                                      dataset='val')

    # wrap your data and label into Tensor
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # train
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, False, args.arch + '-epoch-' + str(args.epochs)+'-checkpoint.pth.tar')


def main_testing():
    global args
    args = parser.parse_args()
    # create model
    model = modifiednet.main(args.arch)
    if use_gpu:
        model.cuda()

    checkpoint = torch.load('./AlexNet-epoch50-checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss()  # loss for classification

    # load train data
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomSizedCrop(input_size),
        ]),
        'val': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomSizedCrop(input_size),
        ])
    }

    # need to define

    num_class = 3

    val_dataset = datasets.customData(img_path='',
                                      txt_path='/home/hongkai/PyTorch_Tutorial/pytorch_MultiTask-two-branches/data/val.txt',
                                      data_transforms=data_transforms,
                                      dataset='val')

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss1_am = AverageMeter()
    # loss2_am = AverageMeter()
    loss_am = AverageMeter()
    # loss2_angle_am = AverageMeter()
    # loss2_point_am = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    accu_no_train = 0
    for i, (input_left, input_right, target_cls) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_gpu:  # transfer to GPU
            input_left_var = Variable(input_left.cuda())
            input_right_var = Variable(input_right.cuda())
            target_cls = Variable(target_cls.cuda())
        else:
            input_left_var = Variable(input_left)
            input_right_var = Variable(input_right)
            target_cls = Variable(target_cls)


        # compute output
        output = model(input_left_var, input_right_var)
        loss = criterion(output, target_cls.data)
        # measure accuracy and record loss
        pred = output.max(1)[1]
        accu_no_train += pred.eq(target_cls.view_as(pred)).sum().item()  # compute the accuracy of classification
        accu_no_train_per = '{:.2%}'.format(accu_no_train / (i + 1))

        loss_am.update(loss.item(), input_left.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Accuracy: {3}\t'                
                  'Loss: {loss.val:.4f}'.format(epoch, i, len(train_loader), accu_no_train_per, loss=loss_am))

        gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_am = AverageMeter()

    class_num = 3
    classes = ('A', 'B', 'C')
    class_correct = list(0. for i in range(class_num))
    class_total = list(0. for i in range(class_num))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    accu_no_val = 0
    # predict_v = torch.from_numpy([])
    for i, (input_left, input_right, target_cls) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # target = target.cuda(async=True)
        if use_gpu:
            input_left_var = Variable(input_left.cuda())
            input_right_var = Variable(input_right.cuda())
            target_cls = Variable(target_cls.cuda())
        else:
            input_left_var = Variable(input_left)
            input_right_var = Variable(input_right)
            target_cls = Variable(target_cls)

        output = model(input_left_var, input_right_var)
        loss = criterion(output, target_cls.data)  # compute two losses
        # measure accuracy and record loss
        pred = output.max(1)[1]
        accu_no_val += pred.eq(target_cls.view_as(pred)).sum().item()
        accu_no_val_per = '{:.4%}'.format(accu_no_val / (i + 1))
        loss_am.update(loss.item(), input_left.size(0))

        c = (pred == target_cls).squeeze()
        for j in range(len(target_cls)):
            label = target_cls[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Accuracy: {2}: \t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(i+1, len(val_loader), accu_no_val_per, loss=loss_am))

    mean_accuracy = 0
    for i in range(class_num):
        print('Final Accuracy of %2s : %.4f %%' % (classes[i], 100 * class_correct[i] / float(class_total[i])))
        mean_accuracy += 100 * class_correct[i] / float(class_total[i])
    print('Final Overall Mean Accuracy: %.4f %%' % (mean_accuracy / float(class_num)))

    return loss.data


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_multiclass(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    # for training and validation
    main()
    # for testing/validation
    # main_testing()