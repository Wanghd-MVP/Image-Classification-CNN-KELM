#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/21 18:44
# @Author : LiFH
# @Site : 
# @File : label_svm.py
# @Software: PyCharm

from sklearn import svm
import os
import numpy as np
from config import opt
from ELMClassifier.random_hidden_layer import RBFRandomHiddenLayer
from ELMClassifier.random_hidden_layer import SimpleRandomHiddenLayer
from ELMClassifier.elm import ELMClassifier
from sklearn.externals import joblib
import os
import numpy as np
from config import opt
import time


if __name__ == '__main__':
    # x_train, y_train = read_vector('./train_label_vector/' + opt.model + '/')
    # print(y_train)
    # x_test, y_test = read_vector('./test_label_vector/'+opt.model + '/')
    #
    train_dict = np.load('npys/' + opt.model + '_label/select.npy').item()
    x_train = train_dict['feature']
    y_train = train_dict['label']
    istrue_train = train_dict['is_true']
    test_dict = np.load('npys/'+opt.model+'_label/test.npy').item()
    x_test, y_test = test_dict['feature'],test_dict['label']
    istrue_test = test_dict['is_true']
    clf = svm.SVC(gamma='auto',kernel='rbf')

    print('开始训练：')
    clf.fit(x_train,istrue_train)

    print('开始测试：')
    result = clf.predict(x_test)
    zero = np.zeros(())
    print(result.size)
    print(np.sum((result==1)))
    print(np.sum((result==0)))
    true_index = np.where(result == 1)
    false_index = np.where(result == 0)[0]

    translabel_x = x_test[false_index]
    translabel_y = y_test[false_index]

    a = np.sum((istrue_test[false_index] == 1)==True)  # 1 0
    b = np.sum((istrue_test[true_index] == 0)==True)   # 0 1
    c = np.sum((istrue_test[false_index] == 0)==True)  # 0 0
    d = np.sum((istrue_test[true_index] == 1)==True)  # 1 1



    print(np.sum((result==istrue_test) == True)/result.size)

    n_hidden = 1000
    siglayer1 = RBFRandomHiddenLayer(n_hidden=n_hidden, gamma=1e-4, use_exemplars=True)
    # siglayer1 = SimpleRandomHiddenLayer(n_hidden=n_hidden, activation_func='sigmoid')
    # 4500 sigmoid 0.741638741047375
    # 4500
    start = time.time()
    clf1 = ELMClassifier(siglayer1)
    clf1.fit(x_train, y_train)
    joblib.dump(clf1, './KELM/' + opt.model + '_KELM_' + str(n_hidden) + '_2.pkl')
    end = time.time()
    print("训练时间", end - start)

    print("开始测试：")
    start = time.time()
    clf1 = joblib.load('./KELM/' + opt.model + '_KELM_' + str(n_hidden) + '_2.pkl')
    pre_result = clf1.predict(translabel_x)
    end = time.time()
    print("测试时间", end - start)
    isTrue = pre_result == translabel_y
    acc = np.sum(isTrue == True) / pre_result.size
    print(acc)
    print(acc - 0.803989)