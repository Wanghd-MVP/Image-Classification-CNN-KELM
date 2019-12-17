#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/15 12:50
# @Author : LiFH
# @Site : 
# @File : elm_train.py
# @Software: PyCharm

import elm
import os,numpy as np
from config import opt
def read_vector(path):
    dir = os.listdir(path)
    #
    train = np.load(path + dir[0])
    x = train['arr_0']
    y = np.expand_dims(train['arr_1'],axis=1)
    data = np.hstack((y,x))
    print("----")
    print(data)
    for item in dir:
        temp = np.load(path + item)

        col = np.hstack((np.expand_dims(temp['arr_1'],axis=1),temp['arr_0']))
        print(col.shape)
        data = np.concatenate((data, col), axis=0)
        print(data.shape)
    return data

if __name__ == '__main__':

    train_data = read_vector('./train_label_vector/'+opt.model+'/')
    # test_data = train_data
    test_data = read_vector('./test_label_vector/'+opt.model+'/')
    # download an example dataset from
    # https://github.com/acba/elm/tree/develop/tests/data


    # load dataset
    # data = elm.read("iris.data")
    # print(data)
    # create a classifier

    # params = ["rbf", 5, []]
    # elmk = elm.ELMKernel(params)
    params = ["rbf", 9, []]
    elmk = elm.ELMKernel()

    # search for best parameter for this dataset
    # define "kfold" cross-validation method, "accuracy" as a objective function
    # to be optimized and perform 10 searching steps.
    # best parameters will be saved inside 'elmk' object
    # elmk.search_param(train_data, cv="kfold", of="accuracy", eval=10)

    # split data in training and testing sets
    # use 80% of dataset to training and shuffle data before splitting
    # tr_set, te_set = elm.split_sets(data, training_percent=.8, perm=True)
    # print(tr_set)
    #train and test
    # results are Error objects
    tr_result = elmk.train(train_data)
    te_result = elmk.test(test_data)
    print(te_result.get_accuracy())