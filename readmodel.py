#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/12 16:25
# @Author : LiFH
# @Site : 
# @File : readmodel.py
# @Software: PyCharm

import torch
from config import opt
model_filename =  opt.dataset+'_'+opt.model+'_best.pth.tar'
state = torch.load(model_filename)
print(state['epoch'])
print(state['best_prec1'])

