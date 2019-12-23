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
# import models
from torchvision import models
from torchnet import meter
from data.dataset import Caltech256
from data.dataset import CIFAR100
from torch.utils.data import DataLoader
import shutil
import time
import numpy as np
from kelm import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
best_prec1 = 0
def main():
    global args, best_prec1
    opt.output()
    if opt.pretrained:
        if opt.model == 'resnet18':
            model = models.resnet18(pretrained=True, num_classes = 1000)
        elif opt.model == 'resnet34':
            model = models.resnet34(pretrained=True, num_classes = 1000)
        elif opt.model == 'resnet50':
            model = models.resnet50(pretrained=False, num_classes=1000)
    else:
        from models import ResNet
        if opt.model == 'resnet18':
            model = ResNet.resnet18()
        elif opt.model == 'resnet34':
            model = ResNet.resnet34()
        elif opt.model == 'resnet50':
            model = ResNet.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, opt.class_num)

    model = nn.DataParallel(model)
    model.cuda()

    start_epoch = 0
    if opt.checkpoint_epochs:
        filename = opt.checkpoints_dir +'/'+str(opt.checkpoint_epochs)+ '.pth.tar'
        pth = torch.load(filename)
        state_dict = pth['state_dict']
        start_epoch = pth['epoch']
        model.load_state_dict(state_dict)

    # Data loading
    if opt.dataset == 'caltech256':
        train_data = Caltech256(opt.data_root, train=True)
        val_data = Caltech256(opt.data_root, train=False)
    elif opt.dataset == 'cifar100':
        train_data = CIFAR100(opt.data_root, train=True)
        val_data = CIFAR100(opt.data_root, train=False)


    train_dataloader = DataLoader(train_data,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data, opt.test_batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                pin_memory=True)

    # define loss function (criterion) and optimizer
    if opt.label_smooth:
        criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, lb_ignore=opt.class_num).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr = opt.lr,
                                momentum=opt.momentum,
                                 weight_decay= opt.weight_decay)


    result = []
    for epoch in range(start_epoch+1,opt.max_epoch):

        if opt.is_adjust_learning_rate:
            adjust_learning_rate(epoch)

        # train for on epoch
        # trainOutput,trainFeature,trainTarget = train(train_dataloader,model,criterion,optimizer,epoch)
        train(train_dataloader,model,criterion,optimizer,epoch)

         # evaluate on validation set
        # prec1,testOutput,testFeature,testTarget = validate(val_dataloader,model,criterion)
        prec1 = validate(val_dataloader,model,criterion)

        label_kelm_prec1 =0

        # if opt.is_label_elm:
        #     label_clf = kelm_train(trainOutput, trainTarget
        #                            ,hidden_layer=opt.label_kernel,n_hidden=opt.label_hidden_node)
        #     label_kelm_prec1 =kelm_test(label_clf,testOutput,testTarget,prec1)
        #
        # feature_kelm_prec1 = 0
        # if opt.is_feature_elm:
        #     feature_clf = kelm_train(trainFeature,trainTarget
        #                              ,hidden_layer=opt.feature_kernel,n_hidden=opt.feature_hidden_node)
        #     feature_kelm_prec1 = kelm_test(feature_clf,testFeature,testTarget,prec1)
        #
        # result.append([prec1,label_kelm_prec1,label_kelm_prec1-prec1,
        #                feature_kelm_prec1,feature_kelm_prec1-prec1])
        result.append([epoch,prec1])
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        print('----当前精确度：',prec1,'----最好精确度-----',best_prec1)
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch':epoch + 1,
            'model': opt.model,
            'state_dict' : model.state_dict(),
            'best_prec1' : prec1,
        },is_best,opt.checkpoints_dir)

        save_everypoint({
            'epoch':epoch + 1,
            'model': opt.model,
            'state_dict' : model.state_dict(),
            'best_prec1' : prec1,
        },opt.checkpoints_dir,epoch + 1)

        write_csv(result,opt.result_file)



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

    trainOutput = []   # label
    trainFeature = []  # feature
    trainTarget = []   # target

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # print(target)

        if opt.use_gpu:
            target = target.cuda(async  = True)
            input_var = torch.autograd.variable(input).cuda()
            target_var = torch.autograd.variable(target).cuda()
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        # compute
        output = model(input_var)

        #
        # remove_fc_model = nn.Sequential(*list(model.children())[:-1])
        # feature = remove_fc_model(input_var).cpu().detach().numpy()
        # feature = feature.reshape(input.size(0), -1)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
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


        # output = output.cpu().detach().numpy()
        #
        #
        # if len(trainOutput)==0:
        #     trainOutput = output
        #     trainTarget = target
        #     trainFeature = feature
        # else:
        #     trainOutput = np.concatenate((trainOutput, output), axis=0)
        #     trainTarget = np.concatenate((trainTarget, target), axis=0)
        #     trainFeature = np.concatenate((trainFeature,feature),axis=0)

    # return trainOutput,trainFeature,trainTarget




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
    #
    # testOutput = []
    # testFeature = []  # feature
    # testTarget = []
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


        # extract the feature
        # remove_fc_model = nn.Sequential(*list(model.children())[:-1])
        # feature = remove_fc_model(input_var).cpu().detach().numpy()
        # feature = feature.reshape(input.size(0), -1)

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

        # output = output.cpu().detach().numpy()
        # if len(testOutput)==0:
        #     testOutput = output
        #     testTarget = target
        #     testFeature = feature
        #     # trainIstrue = isTrue
        # else:
        #     testOutput = np.concatenate((testOutput, output), axis=0)
        #     testTarget = np.concatenate((testTarget, target), axis=0)
        #     testFeature = np.concatenate((testFeature,feature),axis=0)
        #     # trainIstrue = np.concatenate((trainIstrue, isTrue), axis=0)
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    # return top1.avg,testOutput,testFeature,testTarget
    return top1.avg



#
def write_csv(results, file_name):
    import csv
    if os.path.exists(file_name):
        with open(file_name,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cnn','label_cnn_kelm','label_promotion','feature_cnn_kelm','feature_promotion'])
            writer.writerows(results)
    else:
        with open(file_name,'a+') as f:
            writer = csv.writer(f)
            # writer.writerow(['cnn','label_cnn_kelm','label_promotion','feature_cnn_kelm','feature_promotion'])
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

def save_everypoint(state, filename,epoch =0 ):
    if not os.path.exists(filename):
        os.mkdir(filename)
    torch.save(state,filename+'/'+str(epoch)+'.pth.tar')


def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state,filename + '_latest.pth.tar')
    if is_best:
        shutil.copy(filename + '_latest.pth.tar', filename
                    + '_best.pth.tar')



def adjust_learning_rate(epoch):

    if epoch == 30:
        opt.lr = opt.lr *0.01
    elif epoch == 100:
        opt.lr = opt.lr *0.1
    elif epoch == 200:
        opt.lr = opt.lr *0.1
    else:
        print(epoch,opt.lr)

if __name__ == '__main__':
    main()