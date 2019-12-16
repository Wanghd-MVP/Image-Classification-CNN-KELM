#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 12:14
# @Author : LiFH
# @Site : 
# @File : BasicModule.py
# @Software: PyCharm
import torch as t
import time
class BasicModule(t.nn.Module):
    '''
        package the function of nn.Module,
        provide two methods: 'save' and 'load'
    '''
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # the default name of module

    def load(self, path):
        '''
            load the specified model
        :param path:
        :return:
        '''
        self.load_state_dict(t.load(path))

    def save(self, name = None):
        '''
        save module
        :param name:
        :return:
        '''
        if name is None:
            prefix = 'checkpoints/'+ self.model_name + '_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name