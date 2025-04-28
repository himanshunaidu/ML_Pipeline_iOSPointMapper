"""
This backup script contains all the backup Pytorch dataset classes for semantic segmentation.
These are not used in the main pipeline, but are kept for reference and future use.
"""
import os
from PIL import Image
from torch.utils import data
import glob
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose
import numpy as np

class COCOStuffSegmentationBiSeNetv2(data.Dataset):
    """
    COCO Stuff Segmentation Dataset with custom processing for BiSeNetv2 original implementation.
    """
    def __init__(self, root_dir, split='train', year='2017', is_training=True, scale=(0.5, 1.0), crop_size=(513, 513)):
        super(COCOStuffSegmentationBiSeNetv2, self).__init__()
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

        self.train_transforms, self.val_transforms = self.transforms()
        self.is_training = is_training

        ## label mapping, remove non-existing labels
        missing = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]
        remain = [ind for ind in range(182) if not ind in missing]
        self.lb_map = np.arange(256)
        for ind in remain:
            self.lb_map[ind] = remain.index(ind)

    def transforms(self):
        train_transforms = Compose(
            [
                RandomFlip(),
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.crop_size),
                Normalize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.crop_size),
                Normalize()
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

        # Use lb_map to map the labels
        new_mask = np.zeros_like(mask)
        for index, value in enumerate(self.lb_map):
            new_mask[mask == index] = value
        mask = new_mask

        return Image.fromarray(mask)
    
if __name__ == "__main__":
    root_dir = '../../datasets/coco_stuff/coco'

    coco = COCOStuffSegmentationBiSeNetv2(root_dir, split='val', year=2017)
    img, mask = coco.__getitem__(5)
    img.save('rgb.png')
    mask.save('mask.png')