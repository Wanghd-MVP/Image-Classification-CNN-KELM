#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/27 17:34
# @Author : LiFH
# @Site : 
# @File : extract_feature.py
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
from torch.utils.data import DataLoader
import shutil
import time
import numpy as np

def extract_feature(train = True):

    print("------start extract_feature_vector")
    if opt.model == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes = 1000)
    if opt.model == 'resnet50':
        model = models.resnet50(pretrained=False, num_classes = 1000)
    if opt.model == 'resnet34':
        model = models.resnet34(pretrained=False, num_classes = 1000)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 257)
    model.cuda()
    filename = opt.model+'_best.pth.tar'
    pth = torch.load(filename)
    state_dict = pth['state_dict']
    model.load_state_dict(state_dict)
    if train:
        print('---train-vector---')
        data = Caltech256(opt.train_data_root, train=True, test =False )
        dir = 'train_feature/'+opt.model
    else:
        print('---test--vector---')
        data = Caltech256(opt.train_data_root, train=False,test=True)
        dir = 'test_feature/'+opt.model
    dataloader = DataLoader(data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(dataloader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        remove_fc_model = nn.Sequential(*list(model.children())[:-1])
        feature = remove_fc_model(input_var).cpu().detach().numpy()
        feature = feature.reshape(input.size(0), -1)
        print(i)
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savez(dir+"/feature_enc_{}".format(i), feature,target)


if __name__ == '__main__':
    extract_label_feature(True)   # extract the train label
    extract_label_feature(False)  # extract the test label