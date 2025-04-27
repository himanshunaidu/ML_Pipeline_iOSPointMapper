import os
from PIL import Image
from torch.utils import data
import glob
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transforms.semantic_segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose
import numpy as np

cocoStuff_dict = {0:'backgroud', 1:'person', 2:'bicycle', 3:'car', 4:'motorcycle', 6:'bus', 7:'train', 8:'truck',
                  10:'traffic light', 11:'fire hydrant', 12:'street sign', 13:'stop sign', 14:'parking meter',
                  15:'bench', 33: 'suitcase', 41:'skateboard', 64:'potted plant', 92:'banner', 94:'branch',
                  96:'building-other', 97:'bush', 99:'cage', 100:'cardboard', 111:'dirt', 113:'fence', 
                  115:'floor-other', 116:'floor-stone', 124:'grass', 125:'gravel', 126:'ground-other', 128:'house',
                  129:'leaves', 130:'light', 134:'moss', 136:'mud', 140:'pavement', 142:'plant-other', 144:'platform',
                  145:'playfield', 146:'railing', 147:'railroad', 149:'road', 150:'rock', 151:'roof', 154:'sand', 159:'snow',
                  161:'stairs', 162:'stone', 164:'structural-other', 169:'tree', 171: 'wall-brick', 172:'wall-concrete', 
                  173:'wall-other', 174:'wall-panel', 175:'wall-stone', 176:'wall-tile', 177:'wall-wood', 178:'water-other', 182:'wood' }

cos2cocoStuff_dict = {0:149, 1:140, 2:96, 3:173, 4:113, 5:0, 6:10, 7:13, 8:129, 9:124,
                      10:0, 11:1, 12:1, 13:3, 14:8, 15:6, 16:7, 17:2, 18:2, 19:0}

cocoStuff_continuous_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7,
                  10:8, 11:9, 12:10, 13:11, 14:12,
                  15:13, 33:14, 41:15, 64:16, 92:17, 94:18,
                  96:19, 97:20, 99:21, 100:22, 111:23, 113:24, 
                  115:25, 116:26, 124:27, 125:28, 126:29, 128:30,
                  129:31, 130:32, 134:33, 136:34, 140:35, 142:36, 144:37,
                  145:38, 146:39, 147:40, 149:41, 150:42, 151:43, 154:44, 159:45,
                  161:46, 162:47, 164:48, 169:49, 171: 50, 172:50, 
                  173:50, 174:50, 175:50, 176:50, 177:50, 178:51, 182:52 }



class COCOStuffSegmentation(data.Dataset):
    # these are the same as the PASCAL VOC dataset

    def __init__(self, root_dir, split='train', year='2017', is_training=True, scale=(0.5, 1.0), crop_size=(513, 513)):
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

        self.train_transforms, self.val_transforms = self.transforms()
        class_numbers = list(cocoStuff_continuous_dict.values())
        class_numbers = np.array(class_numbers)
        class_numbers = np.unique(class_numbers)
        self.num_classes = len(class_numbers)
        self.is_training = is_training

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

        ##################  For tuning on our custom data
        new_mask = np.zeros_like(mask)
        for k, v in cos2cocoStuff_dict.items():
            new_mask[mask == k] = v
        mask = new_mask
        #################

        # mask = mask + 1 # shift by 1 so that 255 becomes 0 and 0-becomes 1, for coco stuff dataset
        new_mask = np.zeros_like(mask)
        for k, v in cocoStuff_continuous_dict.items():
            new_mask[mask == k] = v

        return Image.fromarray(new_mask)


if __name__ == "__main__":
    root_dir = '../../datasets/coco_stuff/coco'

    coco = COCOStuffSegmentation(root_dir, split='val', year=2017)
    img, mask = coco.__getitem__(1)
    print(np.unique(np.array(mask)))
    img.save('rgb.png')
    mask.save('mask.png')