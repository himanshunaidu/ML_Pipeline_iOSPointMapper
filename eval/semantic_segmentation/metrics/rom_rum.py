import math
import torch
import numpy as np
from skimage.measure import label

class ROMRUM(object):
    """
    Helps to calculate the Region-wise over-segmentation measure (ROM) and region-wise under-segmentation measure (RUM).
    """
    def __init__(self, num_classes=21):
        self.num_classes = num_classes

    def preprocess_inputs(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        _, pred = torch.max(output, 1)

        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()
        
        # shift by 1 so that 255 is 0
        pred += 1
        target += 1

        return pred, target
    
    def get_regions(self, img: torch.ByteTensor):
        """
        Get the number of regions in an image using skimage.
        """
        # detect contiguous regions
        img = img.numpy()
        img = np.squeeze(img)
        img = img.astype(np.uint8)
        regions: np.ndarray = label(img)
        return regions

    def get_rom_rum(self, output, target):
        """
        Calculate the Region-wise over-segmentation measure (ROM) and region-wise under-segmentation measure (RUM).
        """
        pred, target = self.preprocess_inputs(output, target)