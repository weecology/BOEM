from typing import Optional, Callable, Any, Tuple, List, Union
import os
import torch
from torch.utils.data import DataLoader, Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import numpy as np
import random
import json
import cv2

class BreedsDataset(Dataset):
    def __init__(self, 
                 root, 
                 is_train: bool = True,
                 transform=None,
                 is_hier: bool = True,
                 is_source: bool = True,
                 path_yn: bool = False,
                 category: str = 'name',
                 sort: str = 'entity13',
                 sourcefile: str = '_train_source.txt',):
        self.is_hier = is_hier
        self.transform = transform
        self.category = category
        self.img_path = []
        self.path_yn = path_yn
        self.super_label_list = []
        self.class_label_list = []

        if is_train:
            filename = sort + sourcefile
            txt = os.path.join('data', filename)
        else:
            if is_source:
                filename = sort + '_val_source.txt'
            else:
                filename = sort + '_val_target.txt'
            txt = os.path.join('data', filename)

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                id = int(line.split()[1])
                super_id = int(line.split()[2])
                self.class_label_list.append(id)
                self.super_label_list.append(super_id)

        self.targets = self.class_label_list  # Sampler needs to use targets

    def __len__(self):
        return len(self.class_label_list)

    def __getitem__(self, index):

        path = self.img_path[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.is_hier:
            if self.path_yn:
                return sample, self.class_label_list[index], self.super_label_list[index], path
            return sample, self.class_label_list[index], self.super_label_list[index]
        else:
            if self.category == 'name':
                return sample, self.class_label_list[index]    
            else:
                return sample, self.super_label_list[index]
