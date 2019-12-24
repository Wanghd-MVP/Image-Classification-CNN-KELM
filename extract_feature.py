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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def extract_feature(is_train = True):

    print("------start extract_vector------")
    if opt.dataset == 'caltech256':
        print("1")
        if opt.model == 'resnet18':
            model = models.resnet18(pretrained=False, num_classes = 1000)
        if opt.model == 'resnet50':
            model = models.resnet50(pretrained=False, num_classes = 1000)
        if opt.model == 'resnet34':
            model = models.resnet34(pretrained=False, num_classes = 1000)
    elif opt.dataset == 'cifar100':
        print("2")
        from models import ResNet
        if opt.model == 'resnet18':
            model = ResNet.resnet18()
        elif opt.model == 'resnet34':
            model = ResNet.resnet34()
        elif opt.model == 'resnet50':
            model = ResNet.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, opt.class_num)

    # model = nn.DataParallel(model)
    model.cuda()
    filename = opt.model+'label_smooth_latest.pth.tar'
    model_filename =  opt.checkpoints_dir+'_best.pth.tar'

    print(model_filename)
    pth = torch.load(model_filename)
    state_dict = pth['state_dict']

    # print(state_dict)
    model.load_state_dict(state_dict)
    # model['0'].fc = Identity()
    # model.fc = Identity()
    data = data_loading(opt.dataset,is_train)
    dataloader = DataLoader(data,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # switch to evaluate mode
    model.eval()

    train_label_output = []
    train_feature_output = []
    train_target = []
    train_istrue = []
    for i, (input, target) in enumerate(dataloader):
        print(i)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        # compute output

        # extract label feature
        # label_output = model(input_var).cpu().detach().numpy()
        # isTrue = label_output.argmax(axis=1) == target

        # extract feature
        print(nn.Sequential(*list(model.children())))
        remove_fc_model = nn.Sequential(*list(model.children())[:-1])
        print(nn.Sequential(*list(remove_fc_model.children())))
        # model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        # print(nn.Sequential(*list(remove_fc_model.children())))
        feature_output = remove_fc_model(input_var).cpu().detach().numpy()
        # print(feature_output.shape)
        feature_output = feature_output.reshape(input.size(0), -1)
        print(feature_output.shape)

        if len(train_target)==0:
            # train_label_output = label_output
            train_feature_output = feature_output
            train_target = target
            # train_istrue = isTrue
        else:
            print(train_feature_output.shape)
            # train_label_output = np.concatenate((train_label_output, label_output), axis=0)
            train_feature_output = np.concatenate((train_feature_output,feature_output),axis=0)
            train_target = np.concatenate((train_target, target), axis=0)
            # train_istrue = np.concatenate((train_istrue, isTrue), axis=0)

    feature_dict = {}
    feature_dict['label'] = train_label_output
    feature_dict['feature'] = train_feature_output
    feature_dict['target'] = train_target
    feature_dict['is_true'] = train_istrue

    save_npy(feature_dict,'npys/'+opt.checkpoints_dir,is_train)
    print('----------end-------------')

def data_loading(dataset='caltech256', is_train = True):
        # Data loading
    if dataset == 'caltech256':
        return Caltech256(opt.data_root, train=is_train)
    elif dataset == 'cifar100':
        return CIFAR100(opt.data_root, train=is_train)


def save_npy(dict,dir,is_train = True):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if is_train:
        np.save(dir+'/train_feature.npy',dict)
    else:
        np.save(dir+'/test_feature.npy',dict)


if __name__ == '__main__':
    extract_feature(is_train = True)   # extract the train label
    extract_feature(is_train = False)  # extract the test label