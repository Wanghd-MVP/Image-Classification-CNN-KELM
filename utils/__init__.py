#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 10:42
# @Author : LiFH
# @Site : 
# @File : __init__.py.py
# @Software: PyCharm

import torch.nn as nn
from torchvision import models
def init_model(opt):
    if opt.pretrained:
        if opt.model == 'resnet18':
            model = models.resnet18(pretrained=True, num_classes=1000)
        elif opt.model == 'resnet34':
            model = models.resnet34(pretrained=True, num_classes=1000)
        elif opt.model == 'resnet50':
            model = models.resnet50(pretrained=True, num_classes=1000)
        elif opt.model == 'vgg19':
            model = models.vgg19(pretrained=True, num_classes=1000)
        elif opt.model == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=True,num_classes=1000)
        elif opt.model == 'densenet121':
            model = models.densenet121(pretrained=True)
            # model = models.densenet121(pretrained=True,num_classes=1000)
    else:
        from models import ResNet
        if opt.model == 'resnet18':
            model = ResNet.resnet18()
        elif opt.model == 'resnet34':
            model = ResNet.resnet34()
        elif opt.model == 'resnet50':
            model = ResNet.resnet50()
    print(opt.model[:5])
    if opt.model[:5] == 'vgg19':
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, opt.class_num)

    else:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, opt.class_num)
    return model