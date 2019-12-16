#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 16:37
# @Author : LiFH
# @Site : 
# @File : dataset.py
# @Software: PyCharm

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt
class Caltech256(data.Dataset):

    def __init__(self,root, transforms = None, train= True,test = False):
        '''
        main goal: get all address of image
        :param root:
        :param transforms:
        :param train:
        :param test:
        '''

        self.test = test
        paths = [os.path.join(root, path) for path in os.listdir(root)]
        imgs = []
        if self.test:
            for path in paths:
                for i in os.listdir(path)[60:]:
                    imgs.append(os.path.join(path,i))
            self.imgs = imgs
        elif train:
            for path in paths:
                for i in os.listdir(path)[:60]:
                    imgs.append(os.path.join(path,i))
                #imgs += os.listdir(path)[:60]
            self.imgs = imgs
        else:
            for path in paths:
                for i in os.listdir(path)[60:]:
                    imgs.append(os.path.join(path,i))
            self.imgs = imgs

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        if self.test or not train:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        '''
        return a image
        :param self:
        :param index:
        :return:
        '''
        img_path = self.imgs[index]

        label = int(img_path.split('/')[-1][:3]) - 1
        # print(label)
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


