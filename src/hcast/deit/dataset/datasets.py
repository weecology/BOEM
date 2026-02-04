# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


from . import aircraft_seeds
from . import aircraft

from . import birds_seeds
from . import birds

from . import breeds
from . import breeds_seeds

from . import inat21_mini
from . import inat21_mini_seeds

from . import usgs
from . import usgs_seeds

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

        

    if args.data_set == 'AIR-HIER':
        dataset = aircraft.FGVCAircraft_Hier(
            args.data_path,
            is_train=is_train,
            transform=transform,
        )
        nb_classes = [100, 70, 30]


    elif args.data_set == 'AIR-HIER-SUPERPIXEL':
        dataset = aircraft_seeds.FGVCAircraft(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [100, 70, 30]

    elif args.data_set == 'BIRD-HIER':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            random_seed=args.random_seed,
            train=is_train,
        )
        nb_classes = [200, 38, 13]


    elif args.data_set == 'BIRD-HIER-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds_seeds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [200, 38, 13]



    elif args.data_set == 'INAT21-MINI-HIER':
        dataset = inat21_mini.iNat21MiniDataset(
            args.data_path,
            transform=transform,
            is_hier=True,
            is_train=is_train,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'INAT21-MINI-HIER-SUPERPIXEL':
        dataset = inat21_mini_seeds.iNat21MiniDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'USGS':
        dataset = usgs.USGSDataset(
            args.data_path,
            transform=transform,
            is_hier=True,
            is_train=is_train,
        )
        nb_classes = [37, 30, 14]

    elif args.data_set == 'USGS-SUPERPIXEL':
        dataset = usgs_seeds.USGSDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [37, 30, 14]

    
    elif args.data_set == 'BREEDS-HIER-SUPERPIXEL':
        dataset = breeds_seeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort = args.breeds_sort,
            path_yn=args.path_yn,
        )
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]



    elif args.data_set == 'BREEDS-HIER':
        dataset = breeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort = args.breeds_sort,
            path_yn=args.path_yn,
        )
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]
  


    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if 'INAT' in args.data_set:
        t.append(transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
