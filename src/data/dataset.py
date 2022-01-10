import torch
import torch.nn as nn

from torch.utils.data import Dataset

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

from data.aug import TRAIN_AUGMENTATIONS, TEST_AUGMENTATIONS

import numpy as np 

DTYPE = torch.uint8
MEM_DTYPE = np.uint8

"""
Data is not much, load into RAM
"""
class CIFAR(Dataset):

    def __init__(self, x, y, cfg, augment):
        super().__init__()

        print('creating dataset')

        self.cfg = cfg

        self.x = np.transpose(x,[0,2,3,1]) # For albumentations package
        self.y = y

        self.augment = augment
        self.TRAIN_AUGMENTATIONS = TRAIN_AUGMENTATIONS

        self.TEST_AUGMENTATIONS = TEST_AUGMENTATIONS

        self.aug_to_tensor = A.Compose([
                    ToTensorV2(),
                    #A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        
        self.channels = 3

        print('DATASET CREATED', self.x.shape, self.y.shape)


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index, test_augment=False):

        x = self.x[index]
        y = self.y[index]

        # Train Augments
        if self.augment:
            aug = A.Compose(
                self.TRAIN_AUGMENTATIONS
            )
            transformed = aug(image=x)
            x = transformed['image']


        # Test Augments
        if test_augment:
            aug = A.Compose([
                self.TEST_AUGMENTATIONS,
            ])
            transformed = aug(image=x)
            x = transformed['image']

        
        transformed = self.aug_to_tensor(image=x)
        x =  transformed['image']

        return x, y