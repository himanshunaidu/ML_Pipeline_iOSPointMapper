import numpy as np
import torch
import cv2
from skimage.measure import label

class Persello(object):
    """
    Helps to calculate the Persello metric for semantic segmentation.
    """
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
    
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

    def over_segmentation_error(self, output, target):
        """
        Calculate the over-segmentation error of a segmentation.
        """
        pred, target = self.preprocess_inputs(output, target)

        pred_regions = self.get_regions(pred)
        pred_regions_unique_labels = np.unique(pred_regions)
        target_regions = self.get_regions(target)
        target_regions_unique_labels = np.unique(target_regions)

        # calculate the over-segmentation error
        over_segmentation_error = 0

    def under_segmentation_error(self, output, target):
        """
        Calculate the under-segmentation error of a segmentation.
        """
        pred, target = self.preprocess_inputs(output, target)

        pred_regions = self.get_regions(pred)
        pred_regions_unique_labels = np.unique(pred_regions)
        target_regions = self.get_regions(target)
        target_regions_unique_labels = np.unique(target_regions)

        # calculate the under-segmentation error
        under_segmentation_error = 0

        