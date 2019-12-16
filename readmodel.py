#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/12 16:25
# @Author : LiFH
# @Site : 
# @File : readmodel.py
# @Software: PyCharm

import torch
from config import opt
state = torch.load(opt.model +"_best.pth.tar")
print(state['best_prec1'])

