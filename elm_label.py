#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/1 12:46
# @Author : LiFH
# @Site : 
# @File : elm_label.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/25 20:07
# @Author : LiFH
# @Site :
# @File : zhuangzi_elm.py
# @Software: PyCharm
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
    # x_test, y_test = read_vector('./test_label_vector/' + opt.model + '/')

    # x_train, y_train = read_vector('./train_feature/' + opt.model + '/')
    # x_test, y_test = read_vector('./test_feature/' + opt.model + '/')
    #
    train_dict = np.load('npys/' + opt.model + '_label/select.npy').item()
    x_train = train_dict['feature']
    y_train = train_dict['label']
    test_dict = np.load('npys/'+opt.model+'_label/test.npy').item()
    x_test, y_test = test_dict['feature'],test_dict['label']
    print("开始训练：")
    start = time.time()
    n_hidden = 1000

#    siglayer1 = SimpleRandomHiddenLayer(n_hidden = n_hidden,activation_func='sigmoid')

    siglayer1 = RBFRandomHiddenLayer(n_hidden=n_hidden, gamma=1e-4, use_exemplars=True)
    clf1 = ELMClassifier(siglayer1)
    clf1.fit(x_train,y_train)
    joblib.dump(clf1,'./KELM/'+opt.model+'_KELM_label'+str(n_hidden)+'_2.pkl')
    end = time.time()
    print("训练时间",end-start)

    print("开始测试：")
    start = time.time()
    clf1 = joblib.load('./KELM/'+opt.model+'_KELM_label'+str(n_hidden)+'_2.pkl')
    pre_result = clf1.predict(x_test)
    end = time.time()
    print("测试时间", end - start)
    isTrue = pre_result==y_test
    acc = np.sum(isTrue == True) / pre_result.size
    print(acc)