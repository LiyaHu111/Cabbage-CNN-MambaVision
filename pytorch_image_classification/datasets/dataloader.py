from typing import Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
import yacs.config
from torch.utils.data import DataLoader
from pytorch_image_classification import create_collator
from pytorch_image_classification.datasets import create_dataset


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def create_dataloader(config: yacs.config.CfgNode,is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle = True,
            num_workers=config.train.dataloader.num_workers,
            pin_memory=config.train.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.validation.batch_size,
            shuffle = False,
            num_workers=config.validation.dataloader.num_workers,
            pin_memory=config.validation.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        return train_loader, val_loader

    else:
        test_dataset = create_dataset(config, is_train)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.test.batch_size,
            shuffle = False,
            num_workers=config.test.dataloader.num_workers,
            pin_memory=config.test.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)
        return test_loader
