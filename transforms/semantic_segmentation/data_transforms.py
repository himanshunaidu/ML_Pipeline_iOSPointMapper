from torchvision import transforms
import torch
import random
from PIL import Image
import math
import torch
import numpy as np
import numbers
from torchvision.transforms import Pad
from torchvision.transforms import functional as F

# Normalization PARAMETERS for the IMAGENET dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=MEAN, std=STD)

imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd):
        self.alphastd = alphastd
        self.eigval = imagenet_pca['eigval']
        self.eigvec = imagenet_pca['eigvec']

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))

class Normalize(object):
    '''
        Normalize the tensors
    '''
    def __call__(self, rgb_img, label_img=None):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, MEAN, STD) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img

class Identity(object):
    '''
        Identity transform
    '''
    def __call__(self, rgb_img, label_img):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, rgb_img, label_img=None):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, self.mean, self.std) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img

class RandomFlip(object):
    '''
        Random Flipping
    '''
    def __call__(self, rgb_img, label_img):
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb_img, label_img

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb_img, label_img):
        if np.random.random() < self.p:
            return rgb_img, label_img
        # assert rgb_img.shape[:2] == label_img.shape[:2]
        return rgb_img[:, ::-1, :], label_img[:, ::-1]


class RandomScale(object):
    '''
    Random scale, where scale is logrithmic
    '''
    def __init__(self, scale=(0.5, 1.0)):
        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        # rgb_img = rgb_img.resize(new_size, Image.ANTIALIAS)
        rgb_img = rgb_img.resize(new_size, Image.LANCZOS)
        label_img = label_img.resize(new_size, Image.NEAREST)
        return rgb_img, label_img


class RandomCrop(object):
    '''
    Randomly crop the image
    '''
    def __init__(self, crop_size, ignore_idx=255):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        pad_along_w = max(0, int((1 + self.crop_size[0] - w) / 2))
        pad_along_h = max(0, int((1 + self.crop_size[1] - h) / 2))
        # padd the images
        rgb_img = Pad(padding=(pad_along_w, pad_along_h), fill=0, padding_mode='constant')(rgb_img)
        label_img = Pad(padding=(pad_along_w, pad_along_h), fill=self.ignore_idx, padding_mode='constant')(label_img)

        i, j, h, w = self.get_params(rgb_img, self.crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)
        return rgb_img, label_img

class VerticalHalfCrop(object):
    '''
    Crop the image in half vertically and return the half depending on the index
    index = 0 for left half, index = 1 for right half
    '''
    def __init__(self, index=0):
        self.index = index

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        i, j = 0, 0
        if self.index == 0:
            i, j, h, w = 0, 0, h, int(w / 2)
        elif self.index == 1:
            i, j, h, w = 0, int(w / 2), h, int(w / 2)
        
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)
        return rgb_img, label_img   


class RandomResizedCrop(object):
    '''
    Randomly crop the image and then resize it
    '''
    def __init__(self, size, scale=(0.5, 1.0), ignore_idx=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size

        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (
                    math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        crop_size = (int(round(w * random_scale)), int(round(h * random_scale)))

        i, j, h, w = self.get_params(rgb_img, crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)

        # rgb_img = rgb_img.resize(self.size, Image.ANTIALIAS)
        rgb_img = rgb_img.resize(self.size, Image.LANCZOS)
        label_img = label_img.resize(self.size, Image.NEAREST)

        return rgb_img, label_img


class Resize(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, rgb_img, label_img):
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        label_img = label_img.resize(self.size, Image.NEAREST)
        return rgb_img, label_img

class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, rgb_img, label_img):
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            rgb_img = self.adj_brightness(rgb_img, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            rgb_img = self.adj_contrast(rgb_img, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            rgb_img = self.adj_saturation(rgb_img, rate)
        return dict(rgb_img=rgb_img, label_img=label_img)

    def adj_saturation(self, rgb_img, rate):
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = rgb_img.shape
        rgb_img = np.matmul(rgb_img.reshape(-1, 3), M).reshape(shape)/3
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
        return rgb_img

    def adj_brightness(self, rgb_img, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[rgb_img]

    def adj_contrast(self, rgb_img, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[rgb_img]


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
