from typing import Tuple, Union
from PIL import Image
import pathlib
import numpy as np
import os
import torch
import torchvision
import cv2
from torch.utils.data import Dataset


class DatasetCabbage3(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.path = dataset_dir
        self.files = [f for f in os.listdir(dataset_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        curfilename = self.files[idx]
        img = Image.open(os.path.join(self.path, curfilename))
        if self.transform:
            img = self.transform(img)
        if 'Backmoth' in curfilename:
            label = 0
        elif 'Leafminer' in curfilename:
            label = 1
        else:
            label = 2
        img = img.resize((224,224))
        return img, label

class Subset_imagenet(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.transform = None

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)

# def create_dataset(, is_train):
#     if is_train:
#         train_transform = create_transform(config, is_train=True)
#         val_transform = create_transform(config, is_train=False)
#         train_dataset = DatasetCabbage3(config.dataset.train_dataset_dir,transform=train_transform)
#         test_dataset = DatasetCabbage3(config.dataset.test_dataset_dir, transform=val_transform)
#         return train_dataset, test_dataset
#     else:
#         raise ValueError()
#
# def main():
#     train_folder = r'D:/BaiduSyncdisk/pytorch_image_classification-master/data/Cabbage3/train'
#     dataset = DatasetCabbage3(train_folder)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
#     for data, labels in dataloader:
#         print(data)
#
# if __name__ == '__main__':
#     main()