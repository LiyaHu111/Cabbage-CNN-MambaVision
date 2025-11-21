from typing import Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
import yacs.config
from torch.utils.data import DataLoader
from collators import create_collator
from datasets import create_dataset


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def create_dataloader(args, is_train:bool):
    if is_train:

        train_dataset, val_dataset = create_dataset(args, is_train)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle = True,
            num_workers = args.workers,
            pin_memory = args.pin_mem,
            worker_init_fn = worker_init_fn)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle = False,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            worker_init_fn=worker_init_fn)

        return train_loader, val_loader

