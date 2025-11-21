from typing import Tuple, Union
from PIL import Image
import pathlib
import numpy as np
import os
import torch
import torchvision
import yacs.config
import cv2
from torch.utils.data import Dataset

from transforms import create_transform

class DatasetCabbage6(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.path = dataset_dir
        self.files = [f for f in os.listdir(dataset_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        curfilename = self.files[idx]
        image_array = Image.open(os.path.join(self.path, curfilename))
        # image_array = image_array.resize([224, 224])
        # image_array = np.array(img)
        if self.transform:
            image_array = self.transform(image_array)
        if 'No_disease' in curfilename:
            label = 0
        elif 'ring_spot' in curfilename:
            label = 1
        elif 'Downy_Mildew' in curfilename:
            label = 2
        elif 'Black_Rot' in curfilename:
            label = 3
        elif 'Aphid_colony' in curfilename:
            label = 4
        elif 'Alternaria_Leaf_Spot' in curfilename:
            label = 5

        return image_array, label

class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


def create_dataset(args, is_train: bool):
    if args.dataset_name in ['Cabbage6']:
        if is_train:
            train_transform = create_transform(args, is_train=True)
            val_transform = create_transform(args, is_train=False)
            train_dataset = DatasetCabbage6(args.train_dataset_dir,transform=train_transform)
            test_dataset = DatasetCabbage6(args.test_dataset_dir, transform=val_transform)
            return train_dataset, test_dataset
    else:
        raise ValueError()

def main():
    train_folder = r'C:/Users/Administrator/Desktop/MambaVision/data/Cabbage6/train'
    dataset = DatasetCabbage6(train_folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    for data, labels in dataloader:
        print(data)

if __name__ == '__main__':
    main()