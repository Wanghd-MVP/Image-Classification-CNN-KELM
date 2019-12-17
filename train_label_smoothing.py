#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 11:56
# @Author : LiFH
# @Site : 
# @File : train.py.py
# @Software: PyCharm
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from LossFunction.label_smoothing import LabelSmoothSoftmaxCEV1
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
    global args, best_prec1
    print(opt)
    print("=>creating model '{}".format(opt.model))
    #model = models.resnet50(pretrained=True, num_classes=1000)

    if opt.model == 'resnet18':
        model = models.resnet18(pretrained=True, num_classes = 1000)
    if opt.model == 'resnet50':
        model = models.resnet50(pretrained=True, num_classes = 1000)
    if opt.model == 'resnet34':
        model = models.resnet34(pretrained=True, num_classes = 1000)
    #model = torch.load('./models/resnet50.pth')
    #print(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 257)
    model.cuda()


    # Data loading
    train_data = Caltech256(opt.train_data_root, train=True)
    val_data = Caltech256(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data, 16,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, lb_ignore=257).cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr = opt.lr,
                                momentum=opt.momentum,
                                 weight_decay= opt.weight_decay)

    for epoch in range(opt.max_epoch):

        # adjust_learning_rate(epoch)

        # train for on epoch
        train(train_dataloader,model,criterion,optimizer,epoch)

        # evaluate on validation set
        prec1 = validate(val_dataloader,model,criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        print('----当前精确度：',prec1,'----最好精确度-----',best_prec1)
        best_prec1 = max(prec1, best_prec1)


        # if not is_best:
        #     adjust_learning_rate()

        save_checkpoint({
            'epoch':epoch + 1,
            'model': opt.model,
            'state_dict' : model.state_dict(),
            'best_prec1' : prec1,
        },is_best,opt.model.lower()+'label_smooth')



def train(train_loader, model, criterion, optimizer, epoch):
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
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # print(target)

        target = target.cuda(async  = True)
        input_var = torch.autograd.variable(input).cuda()
        target_var = torch.autograd.variable(target).cuda()
        # target_var = target_var.float()
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        # compute
        output = model(input_var)
        # print(output.size(),target_var.size())
        # output = output.long()
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        # print(loss)
        # print(loss.item())
        # print(input.size())
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(),input.size(0))

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
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))




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

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i ,(input, target) in enumerate(val_loader):
        target = target.cuda(async = True)
        input_var = torch.autograd.variable(input).cuda()
        target_var = torch.autograd.variable(target).cuda()
        # target = target
        # input_var = input
        # target_var = target
        # model = model.cpu()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(),input.size(0))
        top1.update(prec1.item(),input.size(0))
        top5.update(prec5.item(),input.size(0))

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



def write_csv(results, file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

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

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state,filename + '_latest.pth.tar')
    if is_best:
        shutil.copy(filename + '_latest.pth.tar', filename
                    + '_best.pth.tar')

def adjust_learning_rate():
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = opt.lr * (0.1 ** (epoch // 30))
    print("adjust learning rate")


    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

def adjust_learning_rate(epoch):

    if epoch == 7:
        opt.lr = opt.lr *0.1
    elif epoch == 20:
        opt.lr = opt.lr *0.1
    elif epoch == 40:
        opt.lr = opt.lr *0.1
    else:
        print(epoch,opt.lr)

if __name__ == '__main__':
    main()