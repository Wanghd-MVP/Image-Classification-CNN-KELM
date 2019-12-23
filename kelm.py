#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/25 20:07
# @Author : LiFH
# @Site : 
# @File : kelm.py
# @Software: PyCharm
from ELMClassifier.random_hidden_layer import RBFRandomHiddenLayer
from ELMClassifier.random_hidden_layer import SimpleRandomHiddenLayer
from ELMClassifier.elm import ELMClassifier
from sklearn.externals import joblib
import os
from config import opt
import time
import numpy as np
from ELMClassifier.label_smoothing_elm import ELMClassifierLabelSmooth
from config import opt
import torch
def main():
    n_hidden = 1000
    hidden_layer = 'rbf'  # or linear

    train_dict = np.load('npys/' + opt.model + '_label/train.npy').item()
    x_train = train_dict['feature']

    y_train = train_dict['label']
    test_dict = np.load('npys/'+opt.model+'_label/test.npy').item()
    x_test, y_test = test_dict['feature'],test_dict['label']
    print("开始训练：")
    start = time.time()

    if hidden_layer == 'rbf':
        siglayer = RBFRandomHiddenLayer(n_hidden=n_hidden, gamma=1e-4, use_exemplars=False)
    else:
        siglayer = SimpleRandomHiddenLayer(n_hidden=n_hidden, activation_func='sigmoid')

    clf = ELMClassifier(siglayer)
    clf.fit(x_train,y_train)
    joblib.dump(clf,'./KELM/'+opt.model+'_KELM_'+str(n_hidden)+'.pkl')
    end = time.time()
    print("训练时间",end-start)

    print("开始测试：")
    start = time.time()
    clf1 = joblib.load('./KELM/'+opt.model+'_KELM_'+str(n_hidden)+'.pkl')
    pre_result = clf1.predict(x_test)
    end = time.time()
    print("测试时间", end - start)
    isTrue = pre_result==y_test
    acc = np.sum(isTrue == True) / pre_result.size *100
    print(acc)


    state = torch.load(opt.model + "label_smooth_latest.pth.tar")
    print("promotion:")
    print(acc-state['best_prec1'])


def kelm_train(x_train,y_train,hidden_layer='rbf',n_hidden = 1000,use_label_smooth=True):
    print("开始训练：")
    start = time.time()
    if hidden_layer == 'rbf':
        siglayer = RBFRandomHiddenLayer(n_hidden=n_hidden, gamma=1e-4, use_exemplars=False)
    elif hidden_layer == 'sigmoid':
        siglayer = SimpleRandomHiddenLayer(n_hidden=n_hidden, activation_func='sigmoid')

    if use_label_smooth:
        print("use_label_smooth:")
        clf = ELMClassifierLabelSmooth(siglayer)
    else:
        clf = ELMClassifier(siglayer)
    clf.fit(x_train, y_train)
    end = time.time()
    print("训练时间", end - start)
    return clf
    # joblib.dump(clf, './KELM/·' + opt.model + '_KELM_' + str(n_hidden) + '.pkl')


def kelm_test(clf ,x_test, y_test,prec1 = 0):
    print("开始测试：")
    start = time.time()
    pre_result = clf.predict(x_test)
    end = time.time()
    print("测试时间", end - start)
    isTrue = pre_result == y_test
    acc = np.sum(isTrue == True) / pre_result.size * 100
    print('精确度',acc)
    print('提升',acc - prec1)
    return acc



if __name__ == '__main__':
    # main()
    dir = 'npys/'+opt.dataset+'_'+opt.model
    # train_filename = dir+'/True.npy'
    # train_dict = np.load(train_filename).item()
    # x_train = train_dict['feature']
    # y_train = train_dict['label']
    # clf = kelm_train(x_train,y_train,'rbf',1000)
    # test_filename = dir+'/False.npy'
    # test_dict = np.load(test_filename).item()
    # x_test = test_dict['feature']
    # y_test = test_dict['label']
    # kelm_test(clf, x_test, y_test,65.75)


    train_filename = dir+'/True.npy'
    train_dict = np.load(train_filename).item()
    x_train = train_dict['feature']
    y_train = train_dict['label']
    clf = kelm_train(x_train,y_train,'rbf',1000,use_label_smooth=False)
    test_filename = dir+'/False.npy'
    test_dict = np.load(test_filename).item()
    x_test = test_dict['feature']
    y_test = test_dict['label']


    model_filename =  opt.dataset+'_'+opt.model+'_best.pth.tar'

    state = torch.load(model_filename)
    kelm_test(clf, x_test, y_test,state['best_prec1'])
    # resnet18  top1(75.4724)
    # n_hidden   gamma   use_examplars  top1                    训练时间
    # 500       1e-5     False          0.7560943557395361    2.686943292617798
    # 800       5e-4     False          0.7615480649188514
    # 800       1e-2     False          0.7611538208817925
    # 800       1e-3     False          0.7630593337275774
    # 900       1e-4     False          0.7633878704251265   4.65047287940979
    # 1000      5e-4     False          0.7653590906104212
    # 1000      1e-3     False          0.7626650896905184   6.939090728759766
    # 1000      1e-5     False          0.7664761153820882   6.168056488037109
    # 1000      1e-4     False          0.7667389447401275   5.482727527618408
    # 1000      5e-5     False          0.7633221630856166   5.589937448501587
    # 1100      1e-4     False          0.7633878704251265   5.421499252319336
    # 1100      1e-3     False          0.7656876273079704   6.1682233810424805
    # 1200      1e-5     False          0.7658190419869899   7.9823219776153564
    # 2000      1e-4     False          0.7652276759314015   13.024249076843262

    #
    # resnet34 77.8495
    # 1000      1e-3    False      0.78809383008082       5.246227741241455
    # 2000      1e-4    False      79.36(1.51)
    # 3000      1e-4    False      79.94(2.09)
    # 4000      1e-4    False      80.23(2.38)

    # resnet50 80.3989(top1)

    # resnet50 80.40%(top1)                  label              feature
    # n_hidden  gamma  use_examplars    top1      训练时间     top1      训练时间
    # 900       1e-3   False            81.54%     4.21s     81.25%     14.10s
    # 900       1e-3   True             80.28%     4.69s     81.24%     19.01s

    # 1000      1e-3   False            81.86%     5.34s     81.33%     18.97s
    # 1000      1e-3   True             80.41%     5.91s     80.93%     19.16s

    # 1100      1e-3   False            81.69%     6.43s     81.45%     20.70s
    # 1100      1e-3   True             80.43%     5.99s     81.36%     20.72s

    # 5000      1e-3   False            81.83%     66.71s    82.50%     180.34s
    # 5000      1e-3   True                                  82.27%     188.59s

    # 10000     1e-3   False                                 81.10%     719.83s
    # elm
    # 5000      sigmoid                                      81.12%      60.49s

    #
    # 4500 sigmoid 0.741638741047375
    # 4500
    # clf1 = ELMClassifierLabelSmooth(siglayer1)
