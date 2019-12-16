#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/20 15:26
# @Author : LiFH
# @Site : 
# @File : isTrue.py
# @Software: PyCharm

import os
import numpy as np
from config import opt


def read_vector(path):
    dir = os.listdir(path)
    train = np.load(path + dir[0])
    x = train['arr_0']
    y = train['arr_1']
    print(x[0].argmax())
    print(y[0])
    isTrue = train['arr_2']
    print(isTrue)
    for item in dir:
        temp = np.load(path + item)
        x = np.concatenate((x, temp['arr_0']), axis=0)
        y = np.concatenate((y, temp['arr_1']), axis=0)
    return x,y

def main():
    x_train ,y_train = read_vector('./train_label_vector/'+opt.model+'/')
    length = y_train.size
    tp = 0  # 分类正确的
    fn = 0  # 分类错误的

    print(x_train.argmax(axis=1)==y_train)
    # for i in range(0, length):
    #     # print(i)
    #     x_test_sort = -np.sort(-x_train[i])
    #     if np.argmax(x_train[i]) == y_train[i]:
    #         print(x_test_sort[0]>4)
    #         tp += 1
    #         # print(max(x_test[i])>5)
    #     else:
    #         fn += 1

    # x_test,y_test = read_vector('./te
    #
    #             # print(x_test_sort[0] - x_test_sort[1]<5)
    #             # print(max(x_test[i]) > 6)
    #             # print(-np.sort(-x_train[i]))
    #     print(tp)
    #     print(tp / length)
    #
    #     print(fn)
    #     print(fn / length)st_label_vector/'+opt.model+'/')
    # length = y_test.size
    # tp = 0 # 分类正确的
    # fn = 0 # 分类错误的
    # for i in range(0,length):
    #     # print(i)
    #     x_test_sort = -np.sort(-x_test[i])
    #     if np.argmax(x_test[i]) == y_test[i]:
    #         tp += 1
    #         # print(max(x_test[i])>5)
    #     else:
    #         fn += 1
    #
    #         print(x_test_sort[0]-x_test_sort[1])
    #         # print(max(x_test[i]) > 6)
    #         #print(-np.sort(-x_train[i]))
    # print(tp)
    # print(tp/length)
    #
    # print(fn)
    # print(fn/length)

if __name__ == '__main__':
    main()