"""Define datasets which generate superpixels."""
from typing import Optional, Callable, Any, Tuple, List, Union
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
#import torchvision.datasets.folder as folder
from torchvision.datasets import VisionDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import PIL.Image
import time

class FGVCAircraft(VisionDataset):
    def __init__(self,
                 root: Union[str, Path],
                 is_train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_hier: bool = True,
                 category: str = 'name',
                 mean: Union[List, Tuple] = IMAGENET_DEFAULT_MEAN,
                 std: Union[List, Tuple] = IMAGENET_DEFAULT_STD,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor=1.0):
        super(FGVCAircraft, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform)
        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor
        self.is_hier = is_hier
        self.category = category

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        f = open('data/Air.csv', 'r', encoding='utf-8-sig')
        lines = f.readlines()
        f.close()
        clsname_to_id = {}
        for line in lines:
            image_name, label_name = line.strip().split(",", 1)
            image_name = image_name.strip('"')
            clsname_to_id[image_name] = label_name

        image_data_folder = os.path.join(self._data_path, "data", "images")
        
        if is_train:
            labels_file = os.path.join(self._data_path, "data", "images_variant_trainval.txt")
        else:
            labels_file = os.path.join(self._data_path, "data", "images_variant_test.txt")

        self._image_files = []
        self._labels = []
        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                self._labels.append(int(clsname_to_id[label_name])-1)


    def __len__(self) -> int:
        return len(self._labels)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)


    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self._image_files[index]
        target = self._labels[index]
        family_target = trees[target][1]-1
        mf_target = trees[target][2]-1
        sample = PIL.Image.open(path).convert("RGB")
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Prepare arguments when multi-view pipeline is adopted.
        compactness = self.compactness
        blur_ops = self.blur_ops
        n_segments = self.n_segments
        scale_factor = self.scale_factor
        if isinstance(sample, (list, tuple)):
            if not isinstance(compactness, (list, tuple)):
                compactness = [compactness] * len(sample)

            if not isinstance(n_segments, (list, tuple)):
                n_segments = [n_segments] * len(sample)

            if not isinstance(blur_ops, (list, tuple)):
                blur_ops = [blur_ops] * len(sample)

            if not isinstance(scale_factor, (list, tuple)):
                scale_factor = [scale_factor] * len(sample)


        # Generate superpixels.
        if isinstance(sample, (list, tuple)):
            segments = []
            for samp, comp, n_seg, blur_op, scale in zip(sample, compactness, n_segments, blur_ops, scale_factor):
                if blur_op is not None:
                    samp = blur_op(samp)
                samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
                samp = (samp * 255).astype(np.uint8)
                samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
                seeds = cv2.ximgproc.createSuperpixelSEEDS(
                    samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                    histogram_bins=5, double_step=False);
                seeds.iterate(samp, num_iterations=15);
                segment = seeds.getLabels()
                segment = torch.LongTensor(segment)
                segments.append(segment)
        else:
            if blur_ops is not None:
                samp = blur_ops(sample)
            else:
              samp = sample
            samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
            samp = (samp * 255).astype(np.uint8)
            samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                histogram_bins=5, double_step=False);
            seeds.iterate(samp, num_iterations=15);
            segments = seeds.getLabels()
            segments = torch.LongTensor(segments)

        if self.is_hier:
            return sample, segments, target, family_target, mf_target
        else:
            if self.category == 'name':
                return sample, segments, target
            elif self.category == 'family':
                return sample, segments, family_target
            else:
                return sample, segments, mf_target


#target, family, order
trees = [
[1, 1, 1],
[2, 2, 1],
[3, 3, 1],
[4, 3, 1],
[5, 3, 1],
[6, 3, 1],
[7, 4, 1],
[8, 4, 1],
[9, 5, 1],
[10, 5, 1],
[11, 5, 1],
[12, 5, 1],
[13, 6, 1],
[14, 7, 2],
[15, 8, 3],
[16, 9, 3],
[17, 10, 7],
[18, 10, 7],
[19, 11, 7],
[20, 12, 4],
[21, 13, 5],
[22, 14, 5],
[23, 15, 5],
[24, 16, 5],
[25, 16, 5],
[26, 16, 5],
[27, 16, 5],
[28, 16, 5],
[29, 16, 5],
[30, 16, 5],
[31, 16, 5],
[32, 17, 5],
[33, 17, 5],
[34, 17, 5],
[35, 17, 5],
[36, 18, 5],
[37, 18, 5],
[38, 19, 5],
[39, 19, 5],
[40, 19, 5],
[41, 20, 5],
[42, 20, 5],
[43, 21, 21],
[44, 22, 14],
[45, 23, 9],
[46, 24, 9],
[47, 25, 9],
[48, 25, 9],
[49, 26, 8],
[50, 27, 8],
[51, 28, 8],
[52, 28, 8],
[53, 29, 12],
[54, 29, 12],
[55, 30, 23],
[56, 31, 14],
[57, 32, 14],
[58, 33, 14],
[59, 34, 23],
[60, 35, 12],
[61, 36, 12],
[62, 37, 12],
[63, 38, 13],
[64, 39, 26],
[65, 40, 15],
[66, 41, 15],
[67, 41, 15],
[68, 41, 15],
[69, 42, 15],
[70, 42, 15],
[71, 43, 15],
[72, 44, 16],
[73, 45, 23],
[74, 46, 22],
[75, 47, 11],
[76, 48, 11],
[77, 49, 18],
[78, 50, 18],
[79, 51, 18],
[80, 52, 6],
[81, 53, 19],
[82, 53, 19],
[83, 54, 7],
[84, 55, 20],
[85, 56, 4],
[86, 57, 21],
[87, 58, 23],
[88, 59, 23],
[89, 59, 23],
[90, 60, 23],
[91, 61, 17],
[92, 62, 25],
[93, 63, 27],
[94, 64, 27],
[95, 65, 28],
[96, 66, 10],
[97, 67, 24],
[98, 68, 29],
[99, 69, 29],
[100, 70, 30]
]
