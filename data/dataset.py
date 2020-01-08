#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/20 16:37
# @Author : LiFH
# @Site : 
# @File : dataset.py
# @Software: PyCharm

from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt
import os
import pickle


import numpy
from torch.utils.data import Dataset

class Caltech256(data.Dataset):

    def __init__(self,root, transforms = None, train= True):
        '''
        main goal: get all address of image
        :param root:
        :param transforms:
        :param train:
        :param test:
        '''

        paths = [os.path.join(root, path) for path in os.listdir(root)]
        imgs = []
        if train:
            for path in paths:
                for i in os.listdir(path)[:60]:
                    imgs.append(os.path.join(path,i))
        else :
            for path in paths:
                for i in os.listdir(path)[60:]:
                    imgs.append(os.path.join(path,i))
                #imgs += os.listdir(path)[:60]
        self.imgs = imgs

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        if train:
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
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



class CIFAR100(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transforms=None,train= True):

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        if train:
            with open(os.path.join(path, 'train'), 'rb') as cifar100:
                self.data = pickle.load(cifar100, encoding='bytes')
            transforms = T.Compose([
                # transforms.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            with open(os.path.join(path, 'test'), 'rb') as cifar100:
                self.data = pickle.load(cifar100, encoding='bytes')
            transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        self.transforms= transforms

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        # print(label)
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))
        from PIL import Image
        image = Image.fromarray(image)
        if self.transforms:
            image = self.transforms(image)
        return image,label

from torchvision import datasets

def CIFAR10(path, transforms=None, train=True):

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if train:
        return datasets.CIFAR10(root=path, train=True, download=False, transform=transform_train)
    else:
        return datasets.CIFAR10(root=path, train=False, download=False, transform=transform_test)



