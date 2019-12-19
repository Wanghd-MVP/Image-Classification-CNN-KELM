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

    # dataset path
    train_data_root = './data/caltech256/256_ObjectCategories'
    test_data_root =  './data/caltech256/256_ObjectCategories'
    load_model_path = 'checkpoints/model.pth'

    # cnn config
    class_num = 257   # label classes
    batch_size = 32
    use_gpu = True
    num_workers = 4
    print_freq = 10

    max_epoch = 50
    lr = 0.0001
    lr_decay = 0.01
    weight_decay = 0.01    # 权重衰减
    momentum = 0.9

    # trick.
    label_smooth = True

    # learning rate
    is_adjust_learning_rate = False


    #  是否需要label  elm
    is_label_elm = True
    label_kernel = 'rbf'
    label_hidden_node = 1000


    # 是否需要基于feature elm
    is_feature_elm = True
    feature_kernel = 'rbf'  # rbf or sigmoid
    feature_hidden_node = 1000

    reuslt_file = model
    if label_smooth:
        reuslt_file += '_ls_'

    reuslt_file += feature_kernel +'_'

    reuslt_file += label_hidden_node
    result_file = '.csv'
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
