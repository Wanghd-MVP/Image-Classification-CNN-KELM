#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/26 16:45
# @Author : LiFH
# @Site : 
# @File : extract_label_vector.py
# @Software: PyCharm
import copy

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
from data.dataset import CIFAR10
from torch.utils.data import DataLoader
import shutil
import time
import numpy as np

from utils import init_model
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def extract_label_feature(is_train = True):

    print("------start extract_label_vector------")

    model = init_model(opt)
    # model = nn.DataParallel(model)
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
        label_output = model(input_var).cpu().detach().numpy()
        isTrue = label_output.argmax(axis=1) == target

        if opt.model[:5] == 'vgg19':
            # remove_fc_model = nn.Sequential(*list(model.children())[:-1])
            # num_features = model.classifier[-1].in_features
            remove_fc_model = copy.deepcopy(model)
            # model.classifier[-1] = nn.Linear(num_features, opt.class_num)
            # remove_fc_model = model
            remove_fc_model.classifier = remove_fc_model.classifier[:-1]
            feature_output = remove_fc_model(input_var).cpu().detach().numpy()
            feature_output = feature_output.reshape(input.size(0), -1)
            print("--------")
            print(feature_output.shape)
        else:
        # extract feature
            remove_fc_model = nn.Sequential(*list(model.children())[:-1])
            feature_output = remove_fc_model(input_var).cpu().detach().numpy()
            feature_output = feature_output.reshape(input.size(0), -1)
            print(feature_output.shape)

        if len(train_label_output)==0:
            train_label_output = label_output
            train_feature_output = feature_output
            train_target = target
            train_istrue = isTrue
        else:
            train_label_output = np.concatenate((train_label_output, label_output), axis=0)
            train_feature_output = np.concatenate((train_feature_output,feature_output),axis=0)
            train_target = np.concatenate((train_target, target), axis=0)
            train_istrue = np.concatenate((train_istrue, isTrue), axis=0)

    feature_dict = {}
    print(train_label_output.shape)
    print(train_feature_output)
    print(train_target)
    print(train_istrue)

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
    elif dataset == 'cifar10':
        return CIFAR10(opt.data_root,train = is_train)


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