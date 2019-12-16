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
from utils import Visualizer
# import models
from torchvision import models
from torchnet import meter
from data.dataset import Caltech256
from torch.utils.data import DataLoader
import shutil
import time
import numpy as np

def main(train = True):

    print("------start extract_label_vector")
    if opt.model == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes = 1000)
    if opt.model == 'resnet50':
        model = models.resnet50(pretrained=False, num_classes = 1000)
    if opt.model == 'resnet34':
        model = models.resnet34(pretrained=False, num_classes = 1000)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 257)
    model.cuda()
    filename = opt.model+'label_smooth_latest.pth.tar'
    pth = torch.load(filename)
    state_dict = pth['state_dict']
    model.load_state_dict(state_dict)
    if train:
        print('---train-vector---')
        data = Caltech256(opt.train_data_root, train=True, test =False )
        # dir = 'train_label_vector/'+opt.model
        dir = opt.model + '_label/train.npy'
    else:
        print('---test--vector---')
        data = Caltech256(opt.train_data_root, train=False,test=True)
        # dir = 'test_label_vector/'+opt.model
        dir = opt.model + '_label/test.npy'
    dataloader = DataLoader(data,
                                  batch_size=opt.batch_size,
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
            # trainOutput = np.append(trainOutput,output[0])
            # trainTarget = np.append(trainTarget,target[0])
            trainOutput = np.concatenate((trainOutput, output), axis=0)
            trainTarget = np.concatenate((trainTarget, target), axis=0)
            trainIstrue = np.concatenate((trainIstrue, isTrue), axis=0)
            print(trainOutput.shape)
            print(trainTarget.shape)
            print(trainIstrue.shape)
        # print(trainTarget)
        # if not os.path.exists(dir):
        #     os.makedirs(dir)
        # np.savez(dir+"/enc_{}".format(i), output,target,isTrue)
    print(trainOutput.shape)
    print(trainTarget.shape)

    feature_dict = {}
    feature_dict['feature'] = trainOutput
    feature_dict['label'] = trainTarget
    feature_dict['is_true'] = trainIstrue
    if not os.path.exists('npys/'+opt.model+'_label'):
        os.makedirs('npys/'+opt.model+'_label')
    np.save('npys/'+dir,feature_dict)

    # np.save('npys/trainfeature_select3_7.npy',feature_dict)

if __name__ == '__main__':
    main(True)   # extract the train label
    main(False)  # extract the test label