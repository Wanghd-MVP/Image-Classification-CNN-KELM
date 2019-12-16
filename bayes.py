#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/14 13:22
# @Author : LiFH
# @Site : 
# @File : bayes.py
# @Software: PyCharm

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from config import opt


def read_vector(path):
    dir = os.listdir(path)
    #


    train = np.load(path + dir[0])
    x = train['arr_0']
    y = train['arr_1']
    for item in dir:
        temp = np.load(path + item)
        x = np.concatenate((x, temp['arr_0']), axis=0)
        y = np.concatenate((y, temp['arr_1']), axis=0)
    return x,y

x_train, y_train = read_vector('./train_label_vector/' + opt.model + '/')
x_test, y_test = read_vector('./test_label_vector/' + opt.model + '/')
model = GaussianNB()

model.fit(x_train, y_train)
print(model)

expected = y_test
predicted = model.predict(x_test)
print(expected)
print(predicted)
acc =metrics.accuracy_score(y_true=expected, y_pred=predicted)
print(acc)
# print(metrics.classification_report(expected, predicted))       # 输出分类信息
# label = list(set(Y))    # 去重复，得到标签类别
# print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息