#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 14:46
# @Author : LiFH
# @Site : 
# @File : config.py
# @Software: PyCharm
import warnings
import torch as t
class DefaultConfig(object):
    env = 'default'  # the env of visdom
    # model = 'resnet18'
    # model = 'resnet34'
    model = 'resnet50'

    train_data_root = './data/caltech256/256_ObjectCategories'
    test_data_root =  './data/caltech256/256_ObjectCategories'
    load_model_path = 'checkpoints/model.pth'


    batch_size = 32
    use_gpu = True
    num_workers = 4
    print_freq = 10

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 50
    lr = 0.001
    lr_decay = 0.01
    weight_decay = 0.01    # 权重衰减
    momentum = 0.9


    def parse(self,kwargs):
        '''
        based dictionary to update the param of config
        :param kwargs:
        :return:
        '''

        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribut %s" %k)
            setattr(self, k, v)
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

opt = DefaultConfig()