#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/10 14:12
# @Author : LiFH
# @Site :
# @File : elmdemo.py
# @Software: PyCharm

# 还有一个hpelm 可以用

import elm

# download an example dataset from
# https://github.com/acba/elm/tree/develop/tests/data


# load dataset
data = elm.read("iris.data")

# create a classifier
elmk = elm.ELMKernel()

# search for best parameter for this dataset
# define "kfold" cross-validation method, "accuracy" as a objective function
# to be optimized and perform 10 searching steps.
# best parameters will be saved inside 'elmk' object
elmk.search_param(data, cv="kfold", of="accuracy", eval=10)

# split data in training and testing sets
# use 80% of dataset to training and shuffle data before splitting
tr_set, te_set = elm.split_sets(data, training_percent=.8, perm=True)

#train and test
# results are Error objects
tr_result = elmk.train(tr_set)
te_result = elmk.test(te_set)
print(te_result.predicted_targets)
print(te_result.expected_targets)
print(te_result.get_accuracy())
