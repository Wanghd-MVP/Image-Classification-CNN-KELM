#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/27 20:17
# @Author : LiFH
# @Site : 
# @File : select_sampler.py
# @Software: PyCharm

import os
import numpy as np
from config import opt
import random
def select_vector(path):
    pass

if __name__ == '__main__':
    train_dict = np.load('npys/' + opt.model + '_label/train.npy').item()
    x_train = train_dict['feature']
    y_train = train_dict['label']
    isTrue = train_dict['is_true']
    print(isTrue)

    train_x_true = []
    train_y_true = []

    train_x_flase = []
    train_y_flase = []

    for i,item in enumerate(isTrue):
        if item==1:
            if len(train_x_true) == 0:
                train_x_true = x_train[i].reshape((1,257))
                train_y_true = [y_train[i]]
            else:
                # trainOutput = np.append(trainOutput,output[0])
                # trainTarget = np.append(trainTarget,target[0])
                train_x_true = np.concatenate((train_x_true, x_train[i].reshape((1,257))), axis=0)
                train_y_true = np.concatenate((train_y_true, [y_train[i]]), axis=0)
        else:
            if len(train_x_flase) == 0:
                train_x_flase = x_train[i].reshape((1,257))
                train_y_flase = [y_train[i]]
            else:
                # trainOutput = np.append(trainOutput,output[0])
                # trainTarget = np.append(trainTarget,target[0])
                train_x_flase = np.concatenate((train_x_flase, x_train[i].reshape((1,257))), axis=0)
                train_y_flase = np.concatenate((train_y_flase, [y_train[i]]), axis=0)
    print(train_x_true.shape)
    print(train_x_flase.shape)
    print(train_y_true.shape)
    print(train_y_flase.shape)


    length = train_x_flase.shape[0]

    x_train = np.concatenate((train_x_true[:length],train_x_flase),axis=0)
    y_train = np.concatenate((train_y_true[:length],train_y_flase),axis=0)
    print(y_train)
    print(np.random.shuffle(y_train))
    print(x_train.shape)

    isTrue = np.zeros((length*2))
    isTrue[:length] = 1
    print(isTrue)
    index = [i for i in range(0,length*2)]
    random.shuffle(index)
    print(isTrue[index])
    isTrue = isTrue[index]
    x_train = x_train[index]
    y_train = y_train[index]

    dict = {}
    dict['feature'] = x_train
    dict['label'] = y_train
    dict['is_true'] = isTrue

    np.save('npys/'+opt.model+'_label/select.npy',dict)




