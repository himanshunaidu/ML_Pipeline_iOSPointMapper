import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose, ToTensor
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from data_loader.semantic_segmentation.ios_point_mapper_scripts.custom_maps import ios_point_mapper_to_cityscapes_dict, ios_point_mapper_to_cocoStuff_custom_35_dict, ios_point_mapper_to_cocoStuff_custom_53_dict

IOS_POINT_MAPPER_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']

ios_point_mapper_dict = {
    0: 'background', 1: 'bicycle', 2: 'bike rack', 3: 'bridge', 4: 'building',
    5: 'bus', 6: 'car', 7: 'dynamic', 8: 'fence', 9: 'ground',
    10: 'guard rail', 11: 'motorcycle', 12: 'parking', 13: 'person',
    14: 'pole', 15: 'rail track', 16: 'rider', 17: 'road',
    18: 'sidewalk', 19: 'sky', 20: 'static',
    21: 'terrain', 22: 'traffic light', 23: 'traffic sign',
    24: 'train', 25: 'truck', 26: 'tunnel',
    27: 'vegetation', 28: 'wall'
}

custom_mapping_dicts = {
    'cityscapes': ios_point_mapper_to_cityscapes_dict,
    '53': ios_point_mapper_to_cocoStuff_custom_53_dict,
    '35': ios_point_mapper_to_cocoStuff_custom_35_dict
}

def get_ios_point_mapper_num_classes(is_custom=False, custom_mapping_dict_key=None):
    """
    Returns the number of classes in the iOSPointMapper dataset.
    If is_custom is True, it returns the number of classes based on the custom mapping dictionary.
    """
    if is_custom:
        assert custom_mapping_dict_key is not None, "Custom mapping dictionary key must be provided when is_custom is True."
        custom_mapping_dict_key = custom_mapping_dict_key if custom_mapping_dict_key is not None else '53'
        # Basic cases
        if custom_mapping_dict_key == '53': return 53
        elif custom_mapping_dict_key == '35': return 35
    # else:
    return len(ios_point_mapper_dict.keys())  # Default number of classes in iOSPointMapper without custom mapping

class iOSPointMapperDataset(data.Dataset):
    """
    Dataset class for Edge Mapping dataset.
    
    Note: While the dataset supports both training and testing, the current implementation
    only supports testing with the Cityscapes classes. 
    """

    def __init__(self, root, train=False, scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255,
                 *, mean=MEAN, std=STD,
                 is_custom=False, custom_mapping_dict_key=None):
        """
        Initialize the dataset class.

        Parameters:
        ----------
        root: str
            Path to the dataset folder.

        train: bool
            Flag to indicate if the dataset is for training or testing.

        scale: tuple
            Tuple containing the scaling range for RandomScale transform.

        size: tuple
            Tuple containing the size of the image.

        ignore_idx: int
            Index to ignore in the ground truth mask.
        """
        self.train = train
        if self.train:
            data_file = os.path.join(root, 'train.txt')
        else:
            data_file = os.path.join(root, 'val.txt')

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                label_img_loc = root + os.sep + line_split[1].rstrip()
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.mean = mean
        self.std = std

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

        self.is_custom = is_custom
        assert not is_custom or custom_mapping_dict_key is not None, "Custom mapping dictionary should be provided when is_custom is True."
        custom_mapping_dict_key = custom_mapping_dict_key if custom_mapping_dict_key is not None else '53'
        self.custom_mapping_dict = custom_mapping_dicts[custom_mapping_dict_key] if custom_mapping_dict_key in custom_mapping_dicts else None

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                # Normalize()
                ToTensor(mean=self.mean, std=self.std)
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                # Normalize()
                ToTensor(mean=self.mean, std=self.std)
            ]
        )
        return train_transforms, val_transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        label_img = self._process_mask(label_img)

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img

    def _process_mask(self, mask):
        mask = np.array(mask, dtype=np.uint8)

        ##################  For tuning on our custom data
        if self.is_custom and self.custom_mapping_dict is not None:
            new_mask = np.zeros_like(mask)
            for k, v in self.custom_mapping_dict.items():
                new_mask[mask == k] = v
            mask = new_mask
        
        return Image.fromarray(mask)


class iOSPointMapperDatasetTest(data.Dataset):
    """
    Dataset class for Edge Mapping dataset (test version).
    
    Note: While the dataset supports both training and testing, the current implementation
    only supports testing with the Cityscapes classes. 
    """

    def __init__(self, root, train=False, scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255):
        """
        Initialize the dataset class.

        Parameters:
        ----------
        root: str
            Path to the dataset folder.

        train: bool
            Flag to indicate if the dataset is for training or testing.

        scale: tuple
            Tuple containing the scaling range for RandomScale transform.

        size: tuple
            Tuple containing the size of the image.

        ignore_idx: int
            Index to ignore in the ground truth mask.
        """
        self.train = train
        if self.train:
            data_file = os.path.join(root, 'train.csv')
        else:
            data_file = os.path.join(root, 'val.csv')

        lines = pd.read_csv(data_file)
        lines['rgb_path'] = lines['rgb'].apply(lambda x: os.path.join(root, x))
        self.images = lines['rgb_path'].tolist()

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                # Normalize()
                ToTensor(mean=self.mean, std=self.std)
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                # Normalize()
                ToTensor(mean=self.mean, std=self.std)
            ]
        )
        return train_transforms, val_transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        # Note: The label image is not actually present in the dataset.
        # Hence, we will have it as a grayscale version of the RGB image, and ignore it while returning.
        label_img = Image.open(self.images[index]).convert('L')

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img
