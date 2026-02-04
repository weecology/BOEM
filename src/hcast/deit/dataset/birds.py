"""Define datasets which generate superpixels."""
from typing import Optional, Callable, Any, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.datasets.folder as folder

from .birds_get_tree_target_2 import *

import random
from collections import defaultdict
class ImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 is_hier: bool = True,
                 category: str = 'name',
                 random_seed: int = 1,
                 train: bool = True):
        super(ImageFolder, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file)

        self.is_hier = is_hier
        self.category = category


    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        order_target = trees[target][1]-1
        family_target = trees[target][2]-1
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_hier:
            return sample, target, family_target, order_target
    
        else:
            if self.category == 'name':
                return sample, target
            elif self.category == 'family':
                return sample, family_target
            else:
                return sample, order_target