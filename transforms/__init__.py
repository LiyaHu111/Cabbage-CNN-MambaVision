from typing import Callable, Tuple

import numpy as np
import torchvision
import yacs.config

from .transforms import (
    CenterCrop,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizeCrop,
    # Resize,
    ToTensor,
)

from .cutout import Cutout, DualCutout
from .random_erasing import RandomErasing


def _get_dataset_stats(args):
    name = args.dataset_name
    if name == 'Cabbage3':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    else:
        raise ValueError()
    return mean, std


def create_transform(args, is_train: bool):
    if args.model_type == 'cabbage':
        return create_cabbage_transform(args, is_train)
    else:
        raise ValueError


def create_cabbage_transform(args, is_train: bool):
    mean, std = _get_dataset_stats(args)
    if is_train:
        transforms = []
        if args.use_random_crop:
            transforms.append(RandomCrop(args))
        if args.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(args))

        transforms.append(Normalize(mean, std))

        if args.use_cutout:
            transforms.append(Cutout(args))
        if args.use_random_erasing:
            transforms.append(RandomErasing(args))
        if args.use_dual_cutout:
            transforms.append(DualCutout(args))

        transforms.append(ToTensor())
    else:
        transforms = [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)

