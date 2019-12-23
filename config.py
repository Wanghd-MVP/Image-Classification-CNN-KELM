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
    model = 'resnet18'
    # model = 'resnet34'
    # model = 'resnet50'
    pretrained = False

    # dataset path
    Caltech256_data_root = './data/caltech256/256_ObjectCategories'
    # CIFAR-100
    Cifar100_data_root = './data/cifar-100-python'

    load_model_path = 'checkpoints/model.pth'

    checkpoint_epochs = 80

    #dataset = 'caltech256'
    dataset = 'cifar100'

    if dataset == 'caltech256':
        data_root = Caltech256_data_root
        class_num = 256

    elif dataset == 'cifar100':
        data_root = Cifar100_data_root
        class_num = 101
    else:
        pass
    # cnn config
    train_batch_size = 128
    test_batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 10

    start_epoch = 0
    max_epoch = 300
    lr = 0.0001
    lr_decay = 0.01
    weight_decay = 0.01    # 权重衰减
    momentum = 0.9

    # trick.
    label_smooth = True

    # learning rate
    is_adjust_learning_rate = False


    #  是否需要label  elm
    is_label_elm = False
    label_kernel = 'rbf'
    label_hidden_node = 1000


    # 是否需要基于feature elm
    is_feature_elm = False
    feature_kernel = 'rbf'  # rbf or sigmoid
    feature_hidden_node = 1000

    result = dataset+'_'+model
    if label_smooth:
        result += '_ls_'

    result += feature_kernel +'_'
    result += str(lr)+'_'
    result += str(label_hidden_node)
    result_file = result  +'.csv'    # training process

    checkpoints_dir =  result #
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

    def output(self):
        print("model:", self.model)
        print("dataset:", self.dataset)
        print('learning_rate:',self.lr)
        print('lable_smoothing:',self.label_smooth)
        print('result_file:',self.result_file)
        print('checkpoint_dir:',self.checkpoints_dir)


opt = DefaultConfig()
