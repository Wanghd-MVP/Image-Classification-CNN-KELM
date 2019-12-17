#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 10:42
# @Author : LiFH
# @Site :
# @File : main.py.py
# @Software: PyCharm

import train
import extract_label_vector
import elm_python

if __name__ == '__main__':
    train.main()
    extract_label_vector.main(True)  # extract the train vector
    extract_label_vector.main(False) # extract the test vector
    elm_python.main()