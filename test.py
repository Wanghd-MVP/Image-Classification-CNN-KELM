#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/26 15:43
# @Author : LiFH
# @Site : 
# @File : test.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from config import opt
from utils import Visualizer
# import models
from torchvision import models
from torchnet import meter
from data.dataset import Caltech256
from torch.utils.data import DataLoader
import shutil
import time

best_prec1 = 0
def main():
    model = models.resnet50(pretrained=True, num_classes=1000)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 257)
    model.cuda()
    filename = 'resnet50_latest.pth.tar'
    pth = torch.load(filename)
    state_dict = pth['state_dict']
    model.load_state_dict(state_dict)
    test_data = Caltech256(opt.train_data_root, train=False, test =True )
    test_dataloader = DataLoader(test_data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_dataloader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update((time.time() - end))
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(test_dataloader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))






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


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i ,(input, target) in enumerate(val_loader):
        target = target.cuda(async = True)
        input_var = torch.autograd.Variable(input, volatile = True).cuda()
        target_var = torch.autograd.Variable(target, volatile = True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0],input.size(0))
        top1.update(prec1[0],input.size(0))
        top5.update(prec5[0],input.size(0))

        # measure elapsed time
        batch_time.update((time.time()-end))
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target,topk=(1,)):
    # compuates the precision@k
    maxk = max(topk)
    batch_size  = target.size(0)


    # print(output)
    _,pred = output.topk(maxk,1,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()