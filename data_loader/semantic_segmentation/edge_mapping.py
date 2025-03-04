import torch
import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose

EDGE_MAPPING_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']

class EdgeMappingSegmentation(data.Dataset):
    """
    Dataset class for Edge Mapping dataset.
    
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
        # Note: The label image is not actually present in the dataset.
        # Hence, we will have it as a grayscale version of the RGB image, and ignore it while returning.
        label_img = Image.open(self.images[index]).convert('L')

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img
