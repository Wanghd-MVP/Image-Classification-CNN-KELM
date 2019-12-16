#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/20 23:05
# @Author : LiFH
# @Site : 
# @File : label_classify.py
# @Software: PyCharm

import torch.nn as nn
import torch
from config import opt
import time
import shutil
import os
import numpy as np
best_prec1 = 0
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model_name = "label_classify"


        #  torch.nn.Linear(in_features, out_features, bias=True)
        self.linear1 = nn.Linear(257, 200)

        self.linear2 = nn.Linear(200,100)

        self.linear3 = nn.Linear(100,10)

        self.linear4 = nn.Linear(10,2)

    def forward(self,x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        #
        x= self.linear2(x)
        x = self.linear3(x)

        x = torch.relu(x)

        x = self.linear4(x)

        return x


def train(path, model, criterion, optimizer, epoch):
    """
    the step of training a network
        1. define the network
        2. define the dataset
        3. define the loss function and optimizer
        4. statistical index
        5. start training
    :return:
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    dir = os.listdir(path)


    train_dict = np.load('npys/' + opt.model + '_label/select.npy').item()
    input = train_dict['feature']

    target = train_dict['is_true']


    i = 0
    for i in range(0,1):

        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async  = True)
        input_var = torch.autograd.variable(input).cuda()
        target =torch.autograd.variable(target).long()
        target_var = torch.autograd.variable(target).cuda()

        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        # compute
        output = model(input_var)
        # print(output.size(),target_var.size())
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var, topk=(1,))
        top1.update(prec1[0], input.shape[0])
        # top5.update(prec5[0],input.size(0))

        # compute gradicent and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                .format(
                epoch, i, 100, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
        i += 1

def validate(test_path, model, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    dir = os.listdir(test_path)

    i = 0
    # for item in dir:
    train_dict = np.load('npys/' + opt.model + '_label/test.npy').item()
    input = train_dict['feature']

    target = train_dict['is_true']


    # measure data loading time
    data_time.update(time.time() - end)

    # target = target.cuda(async  = True)
    input_var = torch.autograd.variable(input).cuda()
    target =torch.autograd.variable(target).long()
    target_var = torch.autograd.variable(target).cuda()

    # input_var = torch.autograd.Variable(input)
    # target_var = torch.autograd.Variable(target)
    # compute
    output = model(input_var)
    # print(output.size(),target_var.size())
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target_var, topk=(1,))
    top1.update(prec1[0], 32)
    # top5.update(prec5[0],input.size(0))

    # compute gradicent and do SGD step


    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % opt.print_freq == 0:
        print('Test: [{0}/{1}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
            i, 100, batch_time=batch_time, loss=losses,
            top1=top1))
    i += 1
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

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


def read_vector(path):
    dir = os.listdir(path)
    train = np.load(path + dir[0])
    x = train['arr_0']
    y = train['arr_1']
    for item in dir:
        temp = np.load(path + item)
        x = np.concatenate((x, temp['arr_0']), axis=0)
        y = np.concatenate((y, temp['arr_1']), axis=0)
    return x,y

def main():
    global  best_prec1
    model = Net().cuda()


    # criterion = nn.MSELoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    epoches = 500
    train_path = './train_label_vector/' + opt.model + '/'
    test_path = './test_label_vector/' + opt.model + '/'
    for epoch in range(epoches):

        train(train_path, model, criterion, optimizer, epoch)

        prec1 = validate(test_path, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        print('----当前精确度：', prec1, '----最好精确度-----', best_prec1)
        best_prec1 = max(prec1, best_prec1)

        # if not is_best:
        #     adjust_learning_rate()

        save_checkpoint({
            'epoch': epoch + 1,
            'model': opt.model+'classify_label',
            'state_dict': model.state_dict(),
            'best_prec1': prec1,
        }, is_best, opt.model.lower()+'classify_label')
def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state,filename + '_latest.pth.tar')
    if is_best:
        shutil.copy(filename + '_latest.pth.tar', filename
                    + '_best.pth.tar')

if __name__ == '__main__':
    main()
