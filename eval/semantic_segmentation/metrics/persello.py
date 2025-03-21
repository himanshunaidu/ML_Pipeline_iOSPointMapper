import numpy as np
import torch
import cv2
from skimage.measure import label

class Persello(object):
    """
    Helps to calculate the Persello metric for semantic segmentation.

    TODO: This code can be optimized by using PyTorch tensors instead of NumPy arrays.

    TODO: Check if the error metric has to take the segmentation labels into account.
    Currently, the error is only region-based, and not label-based.
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

    def get_region_label_dict(self, regions):
        """
        Get a dictionary of region labels along with the index to be used to represent the region label.
        The dictionary will be of the form {region_label: index}.
        Also ignores the background label.
        """
        regions_unique_labels = np.unique(regions)
        regions_unique_labels = regions_unique_labels[regions_unique_labels != 0]
        region_label_dict = {region_label: i for i, region_label in enumerate(regions_unique_labels)}
        return region_label_dict

    def get_intersection(self, pred, target, pred_region, target_object):
        """
        Calculate the intersection of two regions.

        Parameters:
        ----------
        pred: torch.ByteTensor
            The predicted segmentation.

        target: torch.ByteTensor
            The target segmentation.

        pred_region: int
            The region label in the predicted segmentation.

        target_object: int
            The region label in the target segmentation.

        Returns:
        --------
        intersection: int
            The intersection of the two regions.
        """
        intersection = (pred == pred_region) & (target == target_object)
        return np.where(intersection == True, 1, 0)
    
    def get_mod(self, image, label = 1):
        """
        Get the mod of a region in an image.
        """
        return np.sum(np.where(image == label, 1, 0))

    def get_mods_for_regions(self, image, region_labels: dict):
        """
        Get the mod values for all regions in an image.
        """
        mods = []
        for region_label, index in region_labels.items():
            mod = self.get_mod(image, region_label)
            mods.append(mod)
        return mods

    def get_regions_overlap(self, pred_regions, target_regions, 
                            pred_region_labels: dict, target_region_labels: dict):
        """
        Get the overlap between regions in the predicted and target segmentations
        """
        overlap = np.zeros((len(target_region_labels), len(pred_region_labels)))
        for pred_region, i in pred_region_labels.items():
            for target_region, j in target_region_labels.items():
                intersection = self.get_intersection(pred_regions, target_regions, pred_region, target_region)
                overlap[j, i] = self.get_mod(intersection, 1)
        return overlap

    def over_segmentation_error(self, pred_region_labels: dict, target_region_labels: dict, 
                                pred_mods, target_mods, regions_overlap, region_matches):
        """
        Calculate the over-segmentation error of a segmentation.
        """
        oversegmentation_errors = np.zeros(len(target_region_labels))
        for target_label, j in target_region_labels.items():
            oversegmentation_errors[j] = 1 - (regions_overlap[j, region_matches[j]]/target_mods[j])
        return oversegmentation_errors

    def under_segmentation_error(self, pred_region_labels: dict, target_region_labels: dict, 
                                pred_mods, target_mods, regions_overlap, region_matches):
        """
        Calculate the under-segmentation error of a segmentation.
        """
        undersegmentation_errors = np.zeros(len(target_region_labels))
        for target_label, j in target_region_labels.items():
            undersegmentation_errors[j] = 1 - (regions_overlap[j, region_matches[j]]/pred_mods[region_matches[j]])
        return undersegmentation_errors

    def get_persello(self, output, target):
        """
        Calculate the Persello metric for semantic segmentation.
        """
        pred, target = self.preprocess_inputs(output, target)

        # Get the regions in the predicted and target segmentations
        # and ignore the background label
        pred_regions = self.get_regions(pred)
        pred_region_labels = self.get_region_label_dict(pred_regions)
        target_regions = self.get_regions(target)
        target_region_labels = self.get_region_label_dict(target_regions)

        # Get region mod values (including overlap)
        pred_mods = self.get_mods_for_regions(pred_regions, pred_region_labels)
        target_mods = self.get_mods_for_regions(target_regions, target_region_labels)
        regions_overlap = self.get_regions_overlap(pred_regions, target_regions, 
                                                   pred_region_labels, target_region_labels)
        region_matches = np.argmax(regions_overlap, axis=1)

        # Calculate the Persello metrics
        oversegmentation_errors = self.over_segmentation_error(
            pred_region_labels, target_region_labels,
            pred_mods, target_mods, regions_overlap, region_matches)
        undersegmentation_errors = self.under_segmentation_error(
            pred_region_labels, target_region_labels,
            pred_mods, target_mods, regions_overlap, region_matches)
        return oversegmentation_errors, undersegmentation_errors


        