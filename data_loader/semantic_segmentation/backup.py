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
from transforms.semantic_segmentation.data_transforms import \
    RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose, Identity
import numpy as np

"""
TODO: Elimiate the usage of this class.
Currently, it is used extensively by the testing scripts. 
We can simply replace this with the CityscapesSegmentation class by providing custom mean and std.
"""
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