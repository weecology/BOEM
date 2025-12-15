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

class iNat21MiniDataset(Dataset):
    def __init__(self, 
                 root, 
                 is_train: bool = True,
                 transform=None,
                 is_hier: bool = True,
                 path_yn: bool = False,
                 category: str = 'name',):
        self.is_hier = is_hier
        self.transform = transform
        self.category = category
        self.img_path = []
        self.path_yn = path_yn
        self.species_label_list = []
        self.family_label_list = []
        self.order_label_list = []

        if is_train:
            filename = 'data/inat21_mini_train.txt'
        else:
            filename = 'data/inat21_mini_val.txt'

        with open(filename) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                id = int(line.split()[1])
                family_id = int(line.split()[2])
                order_id = int(line.split()[3])
                self.species_label_list.append(id)
                self.family_label_list.append(family_id)
                self.order_label_list.append(order_id)

        self.targets = self.species_label_list  # Sampler needs to use targets

    def __len__(self):
        return len(self.species_label_list)

    def __getitem__(self, index):

        path = self.img_path[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.is_hier:
            if self.path_yn:
                return sample, self.species_label_list[index], self.family_label_list[index], self.order_label_list[index], path
            return sample, self.species_label_list[index], self.family_label_list[index], self.order_label_list[index]
        else:
            if self.category == 'name':
                return sample, self.species_label_list[index]    
            elif self.category == 'family':
                return sample, self.family_label_list[index]
            elif self.category == 'order':
                return sample, self.order_label_list[index]
