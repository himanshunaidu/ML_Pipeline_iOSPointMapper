import torch
import random
from PIL import Image

from transforms.semantic_segmentation.data_transforms import *

class TransformationTrainESPNetv2(object):
    def __init__(self, scale, size, *,
                 mean=MEAN, std=STD):
        self.train_transforms = Compose(
            [
                RandomScale(scale=scale),
                RandomCrop(crop_size=size),
                RandomFlip(),
                ToTensor(mean=mean, std=std)
            ]
        )

    def __call__(self, rgb_img, label_img):
        rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        return rgb_img, label_img
    
class TransformationValESPNetv2(object):
    def __init__(self, size, *, 
                 mean=MEAN, std=STD):
        self.val_transforms = Compose(
            [
                Resize(size=size),
                ToTensor(mean=mean, std=std)
            ]
        )

    def __call__(self, rgb_img, label_img):
        rgb_img, label_img = self.val_transforms(rgb_img, label_img)
        return rgb_img, label_img

class TransformationTrainBiSeNetv2(object):

    def __init__(self, size, scale, *, 
                 mean=MEAN, std=STD, ):
        self.trans_func = Compose([
            RandomResizedCrop(size=size, scale=scale),
            RandomHorizontalFlip(),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
            ToTensor(mean=mean, std=std)
        ])

    def __call__(self, rgb_img, label_img):
        rgb_img, label_img = self.trans_func(rgb_img, label_img)
        return rgb_img, label_img

class TransformationValBiSeNetv2(object):

    def __init__(self, *,
                 mean=MEAN, std=STD):
        self.val_func = Compose([
            ToTensor(mean=mean, std=std)
        ])

    def __call__(self, rgb_img, label_img):
        rgb_img, label_img = self.val_func(rgb_img, label_img)
        return rgb_img, label_img