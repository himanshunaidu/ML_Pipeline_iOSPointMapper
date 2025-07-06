import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose
from transforms.semantic_segmentation.data_transforms import VerticalHalfCrop, Identity, ToTensor
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from data_loader.semantic_segmentation.cityscapes_scripts.process_cityscapes import Label, labels

CITYSCAPE_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']

CITYSCAPE_TRAIN_CMAP = {
    label.trainId: label.color for index, label in enumerate(labels)
}

# Mapping from cityscapes classes to **custom** cocostuff classes
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
cityscape_to_custom_cocoStuff_dict = {0:41, 1:35, 2:19, 3:50, 4:24, 5:0, 6:8, 7:11, 8:31, 9:27,
                            10:0, 11:1, 12:1, 13:3, 14:12, 15:5, 16:6, 17:2, 18:2, 19:0}

custom_mapping_dicts = {
    '53': cityscape_to_custom_cocoStuff_dict
}

def get_cityscapes_num_classes(is_custom=False, custom_mapping_dict_key=None):
    """
    Returns the number of classes in the Cityscapes dataset.
    If is_custom is True, it returns the number of classes based on the custom mapping dictionary.
    """
    if is_custom:
        assert custom_mapping_dict_key is not None, "Custom mapping dictionary key must be provided when is_custom is True."
        custom_mapping_dict_key = custom_mapping_dict_key if custom_mapping_dict_key is not None else '53'
        # Basic cases
        if custom_mapping_dict_key == '53': return 53
    # else:
    return len(CITYSCAPE_CLASS_LIST)  # Default number of classes in Cityscapes without custom mapping

class CityscapesSegmentation(data.Dataset):

    def __init__(self, root, train=True, scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255, coarse=True,
                 *, mean=MEAN, std=STD,
                 is_custom=False, custom_mapping_dict_key=None):
        """
        Note: The split argument was added only recently. Will need to be incorporated more properly. 
        """

        self.train = train
        if self.train:
            data_file = os.path.join(root, 'train.txt')
            if coarse:
                coarse_data_file = os.path.join(root, 'train_coarse.txt')
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

        # if you want to use Coarse data for training
        if train and coarse and os.path.isfile(coarse_data_file):
            with open(coarse_data_file, 'r') as lines:
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

class CityscapesSegmentationForAccessibility(CityscapesSegmentation):
    """
    A class to load the Cityscapes dataset for semantic segmentation with accessibility focus.

    Currently, for accessibility focus, the dataset only splits each image vertically into two halves.
    We assume that both the halves can give a better focus from a pedestrian's perspective.
    """
    def __init__(self, root, train=True, scale=(0.5, 2.0), size=(512, 512), ignore_idx=255, coarse=True,
                 *, mean=MEAN, std=STD,
                 is_custom=False, custom_mapping_dict_key=None):
        super().__init__(root, train, scale, size, ignore_idx, coarse, mean=mean, std=std,
                         is_custom=is_custom, custom_mapping_dict_key=custom_mapping_dict_key)

    def __len__(self):
        # Return twice the number of images for accessibility focus
        return len(self.images) * 2
    
    def __getitem__(self, index):
        # Determine which half of the image to load
        half_index = index // 2
        half_side = index % 2

        rgb_img = Image.open(self.images[half_index]).convert('RGB')
        label_img = Image.open(self.masks[half_index])

        label_img = self._process_mask(label_img)

        # Apply the vertical half crop transformation
        rgb_img, label_img = VerticalHalfCrop(index=half_side)(rgb_img, label_img)
        # Apply the transformations
        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img