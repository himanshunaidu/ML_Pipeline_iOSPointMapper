import torch
import torch.utils.data as data
import os
from PIL import Image
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose
from transforms.semantic_segmentation.data_transforms import VerticalHalfCrop, Identity

CITYSCAPE_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']


class CityscapesSegmentation(data.Dataset):

    def __init__(self, root, train=True, scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255, coarse=True):
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

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                Normalize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                Normalize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img

class CityscapesSegmentationTest(data.Dataset):

    def __init__(self, root, scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255, coarse=True, split = 'train'):
        """
        Note: The split argument was added only recently. Will need to be incorporated more properly. 
        """

        self.train = split == 'train'
        if self.train:
            data_file = os.path.join(root, 'train.txt')
            if coarse:
                coarse_data_file = os.path.join(root, 'train_coarse.txt')
        else:
            data_file = os.path.join(root, "{}.txt".format(split))

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
        if self.train and coarse and os.path.isfile(coarse_data_file):
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

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                Identity()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
                Identity()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img

class CityscapesSegmentationForAccessibility(CityscapesSegmentation):
    """
    A class to load the Cityscapes dataset for semantic segmentation with accessibility focus.

    Currently, for accessibility focus, the dataset only splits each image vertically into two halves.
    We assume that both the halves can give a better focus from a pedestrian's perspective.
    """
    def __init__(self, root, train=True, scale=(0.5, 2.0), size=(512, 512), ignore_idx=255, coarse=True):
        super().__init__(root, train, scale, size, ignore_idx, coarse)

    def __len__(self):
        # Return twice the number of images for accessibility focus
        return len(self.images) * 2
    
    def __getitem__(self, index):
        # Determine which half of the image to load
        half_index = index // 2
        half_side = index % 2

        rgb_img = Image.open(self.images[half_index]).convert('RGB')
        label_img = Image.open(self.masks[half_index])

        # Apply the vertical half crop transformation
        rgb_img, label_img = VerticalHalfCrop(index=half_side)(rgb_img, label_img)
        # Apply the transformations
        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img