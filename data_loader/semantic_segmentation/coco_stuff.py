import os
from PIL import Image
from torch.utils import data
import glob
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose, ToTensor
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from data_loader.semantic_segmentation.cocostuff_scripts.custom_maps import cocoStuff_continuous_53_dict, cocoStuff_continuous_35_dict, cocoStuff_cityscapes_dict, \
    cocoStuff_continuous_11_dict, cocoStuff_continuous_9_dict, cocoStuff_continuous_35_dict_deprecated
import numpy as np

# Custom mapping dictionary for COCOStuff
custom_mapping_dicts = {
    'city': cocoStuff_cityscapes_dict,
    '53': cocoStuff_continuous_53_dict,
    '35': cocoStuff_continuous_35_dict,
    '11': cocoStuff_continuous_11_dict,
    '9': cocoStuff_continuous_9_dict,
    '35_deprecated': cocoStuff_continuous_35_dict_deprecated
}

def get_cocoStuff_num_classes(is_custom=False, custom_mapping_dict_key=None):
    """
    Returns the number of classes in the COCOStuff dataset.
    If is_custom is True, it returns the number of classes based on the custom mapping dictionary.
    """
    if is_custom:
        assert custom_mapping_dict_key is not None, "Custom mapping dictionary key must be provided when is_custom is True."
        custom_mapping_dict_key = custom_mapping_dict_key if custom_mapping_dict_key is not None else '53'
        # Basic cases
        if custom_mapping_dict_key == '53': return 53
        elif custom_mapping_dict_key == '35': return 35
        elif custom_mapping_dict_key == 'city': return 19  # For cityscapes mapping
        elif custom_mapping_dict_key == '11': return 11
        elif custom_mapping_dict_key == '9': return 9
        elif custom_mapping_dict_key == '35_deprecated': return 35  # Deprecated version
    # else:
    return 182  # Default number of classes in COCOStuff without custom mapping

def get_cocoStuff_mean_std():
    """
    Returns the mean and standard deviation values for COCOStuff dataset.
    """
    mean = (0.46962251, 0.4464104,  0.40718787)
    std = (0.27469736, 0.27012361, 0.28515933)
    return mean, std

class COCOStuffSegmentation(data.Dataset):
    # these are the same as the PASCAL VOC dataset

    def __init__(self, root_dir, split='train', year='2017', is_training=True, scale=(0.5, 1.0), 
                 crop_size=(513, 513), ignore_idx=255, 
                 *, mean=MEAN, std=STD,
                 is_custom=False, custom_mapping_dict_key=None):
        super(COCOStuffSegmentation, self).__init__()
        self.img_dir = os.path.join(root_dir, 'images/{}{}'.format(split, year))
        self.annot_dir = os.path.join(root_dir, 'annotations/{}{}'.format(split, year))

        image_list = glob.glob(self.img_dir + os.sep + '*.jpg')
        annotation_list = glob.glob(self.annot_dir + os.sep + '*.png')

        assert len(image_list) == len(annotation_list)

        self.image_list = image_list
        self.annotation_list = annotation_list

        self.split = split

        if isinstance(crop_size, tuple):
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.mean = mean
        self.std = std

        self.train_transforms, self.val_transforms = self.transforms()

        self.is_custom = is_custom
        assert not is_custom or custom_mapping_dict_key is not None, "Custom mapping dictionary should be provided when is_custom is True."
        custom_mapping_dict_key = custom_mapping_dict_key if custom_mapping_dict_key is not None else '53'
        self.custom_mapping_dict = custom_mapping_dicts[custom_mapping_dict_key]

        # class_numbers = list(self.custom_mapping_dict.values())
        # class_numbers = np.array(class_numbers)
        # class_numbers = np.unique(class_numbers)
        # self.num_classes = len(class_numbers)
        self.is_training = is_training

    def transforms(self):
        train_transforms = Compose(
            [
                RandomFlip(),
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.crop_size),
                # Normalize()
                ToTensor(mean=self.mean, std=self.std)
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.crop_size),
                # Normalize()
                ToTensor(mean=self.mean, std=self.std)
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        rgb_img_loc = self.image_list[index]
        annotation_loc = rgb_img_loc.replace('images', 'annotations')
        annotation_loc = annotation_loc.replace('.jpg', '.png')
        rgb_img = Image.open(rgb_img_loc).convert('RGB')
        label_img = Image.open(annotation_loc)

        label_img = self._process_mask(label_img)

        if self.is_training:
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
        #################

        # mask = mask + 1 # shift by 1 so that 255 becomes 0 and 0-becomes 1, for coco stuff dataset
        # new_mask = np.zeros_like(mask)
        # for k, v in cocoStuff_continuous_dict.items():
        #     new_mask[mask == k] = v

        return Image.fromarray(new_mask)

if __name__ == "__main__":
    root_dir = '../../datasets/coco_stuff/coco'

    coco = COCOStuffSegmentation(root_dir, split='val', year=2017)
    # img, mask = coco.__getitem__(5)
    # img.save('rgb.png')
    # mask.save('mask.png')
    unique_mask_values = set()
    for i in range(1000):
        img, mask = coco.__getitem__(i)
        mask = np.array(mask)
        unique_mask_values.update(np.unique(mask))
    print(unique_mask_values)
    print(len(unique_mask_values))