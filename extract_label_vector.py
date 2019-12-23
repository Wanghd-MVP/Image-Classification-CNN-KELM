#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/26 16:45
# @Author : LiFH
# @Site : 
# @File : extract_label_vector.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
from tqdm import tqdm
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

def extract_label_feature(is_train = True):

    print("------start extract_label_vector------")
    if opt.dataset == 'caltech256':
        if opt.model == 'resnet18':
            model = models.resnet18(pretrained=False, num_classes = 1000)
        if opt.model == 'resnet50':
            model = models.resnet50(pretrained=False, num_classes = 1000)
        if opt.model == 'resnet34':
            model = models.resnet34(pretrained=False, num_classes = 1000)
    elif opt.dataset == 'cifar100':
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
    # filename = opt.model+'label_smooth_latest.pth.tar'
    model_filename =  opt.checkpoints_dir+'_best.pth.tar'

    print(model_filename)
    pth = torch.load(model_filename)
    state_dict = pth['state_dict']

    # print(state_dict)
    model.load_state_dict(state_dict)
    data = data_loading(opt.dataset,is_train)
    dataloader = DataLoader(data,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # switch to evaluate mode
    model.eval()

    trainOutput = []
    trainTarget = []

    for i, (input, target) in enumerate(dataloader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        # compute output
        output = model(input_var).cpu().detach().numpy()
        isTrue = output.argmax(axis=1) == target

        if len(trainOutput)==0:
            trainOutput = output
            trainTarget = target
            trainIstrue = isTrue
        else:
            trainOutput = np.concatenate((trainOutput, output), axis=0)
            trainTarget = np.concatenate((trainTarget, target), axis=0)
            trainIstrue = np.concatenate((trainIstrue, isTrue), axis=0)
    feature_dict = {}
    feature_dict['feature'] = trainOutput
    feature_dict['label'] = trainTarget
    feature_dict['is_true'] = trainIstrue
    # save_dir = opt.dataset+'_'+opt.model
    # save_filename = save_dir+'/'+str(is_train)+'.npy'
    # if not os.path.exists('npys/'+save_dir):
    #     os.makedirs('npys/'+save_dir)
    # np.save('npys/'+save_filename,feature_dict)
    save_npy(feature_dict,'npys/'+opt.checkpoints_dir,is_train)
    print('----------end-------------')
    # np.save('npys/trainfeature_select3_7.npy',feature_dict)

def data_loading(dataset='caltech256', is_train = True):
        # Data loading
    if opt.dataset == 'caltech256':
        return Caltech256(opt.data_root, train=is_train)
    elif opt.dataset == 'cifar100':
        return CIFAR100(opt.data_root, train=is_train)


def save_npy(dict,dir,is_train = True):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if is_train:
        np.save(dir+'/train.npy',dict)
    else:
        np.save(dir+'/test.npy',dict)


if __name__ == '__main__':
    extract_label_feature(is_train = True)   # extract the train label
    extract_label_feature(is_train = False)  # extract the test label