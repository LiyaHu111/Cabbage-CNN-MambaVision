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
    Resize,
    ToTensor,
)

from .cutout import Cutout, DualCutout
from .random_erasing import RandomErasing


def _get_dataset_stats(config: yacs.config.CfgNode) -> Tuple[np.ndarray, np.ndarray]:
    name = config.dataset.name
    if name == 'Cabbage6':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    else:
        raise ValueError()
    return mean, std


def create_transform(config: yacs.config.CfgNode, is_train: bool) -> Callable:
    if config.model.type == 'cabbage':
        return create_cabbage_transform(config, is_train)
    else:
        raise ValueError


def create_cabbage_transform(config: yacs.config.CfgNode, is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.use_dual_cutout:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)

