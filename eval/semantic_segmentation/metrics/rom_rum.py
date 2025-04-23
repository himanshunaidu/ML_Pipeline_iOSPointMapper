import math
import torch
from torch import Tensor, ByteTensor
import numpy as np
from skimage.measure import label

class ROMRUM(object):
    """
    Helps to calculate the Region-wise over-segmentation measure (ROM) and region-wise under-segmentation measure (RUM).

    Parameters:
    -----------
    num_classes: int
        The number of classes in the segmentation.

    max_regions: int
        The maximum number of regions that can be practically present in the segmentation.
        (Will ignore regions beyond this number)
    """
    def __init__(self, num_classes=21, max_regions=255, is_output_probabilities=True):
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.is_output_probabilities = is_output_probabilities

    def preprocess_inputs(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        if self.is_output_probabilities:
            _, pred = torch.max(output, 1)
        else:
            pred = output

        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()
        
        pred = pred.type(torch.ByteTensor).clone()
        target = target.type(torch.ByteTensor).clone()
        
        # shift by 1 so that 255 is 0
        # TODO: Check if this is necessary in the current implementation
        #   It seems to be so only because of the way the cityscapes training is being done.
        #   The background label is being given a value of 255, when by convention, it should be 0.
        pred += 1
        target += 1

        return pred, target
    
    def get_regions(self, img: torch.ByteTensor, background_label=0):
        """
        Get the number of regions in an image using skimage.
        # NOTE: Follows the assumption that there are at maximum 255 regions in the segmentation
        """
        # detect contiguous regions
        img_numpy = img.numpy()
        img_numpy = img_numpy.astype(np.uint8)
        regions = label(img_numpy, background=background_label)
        # Truncate the regions to the maximum number of regions
        regions[regions > self.max_regions] = self.max_regions
        return torch.from_numpy(regions)

    def get_region_class_map(self, img: Tensor, regions: Tensor, region_bin_counts: Tensor):
        """
        Get the mapping of region labels to class labels.
        Do this by getting the pixel value of the image at the region label.

        The region_bin_counts tensor is used to get the number of regions in the segmentation.
        """
        region_class_map = torch.zeros(region_bin_counts.shape[0], dtype=torch.long)
        for i in range(1, region_bin_counts.shape[0]):
            region_class_map[i] = img[regions == i][0]
        return region_class_map
    
    def get_regions_overlap(self, pred_regions: Tensor, target_regions: Tensor,
                            pred_region_bin_counts: Tensor, target_region_bin_counts: Tensor):
        """
        Get the overlap between regions in the predicted and target segmentations.

        Instead of performing a nested loop to get the overlap between each region in pred_regions
        with each region in target_regions, we can use the following algorithm:
        - Get a new tensor where each index gives us the value at the same index in pred_regions and target_regions.
        - Get the bin counts of the indices tensor.
        - Reshape the indices_bin_counts tensor to get the overlap tensor.

        Takes advantage of the fact that the region labels are contiguous and start from 0 (0 is the background label).
        
        Parameters:
        ----------
        pred_regions: Tensor
            The regions in the predicted segmentation.

        target_regions: Tensor
            The regions in the target segmentation. Should be the same shape as pred_regions.

        pred_region_bin_counts: Tensor
            The bin counts of the predicted regions.

        target_region_bin_counts: Tensor
            The bin counts of the target regions.

        Returns:
        --------
        overlap: Tensor
            A 2D matrix that gives the size of the intersection between each region in pred_regions
            with each region in target_regions.
            Dimensions: (num_target_regions, num_pred_regions)
        """
        pred_regions_flat = pred_regions.ravel()
        target_regions_flat = target_regions.ravel()

        # Get a new tensor where each index gives us the value at the same index in pred_regions_flat and target_regions_flat
        indices = target_regions_flat * pred_region_bin_counts.shape[0] + pred_regions_flat

        # Get the bin counts of the indices tensor
        indices_bin_counts = torch.bincount(indices, minlength=target_region_bin_counts.shape[0] * pred_region_bin_counts.shape[0])

        # Reshape the indices_bin_counts tensor to get the overlap tensor
        overlap = indices_bin_counts.reshape(target_region_bin_counts.shape[0], pred_region_bin_counts.shape[0])
        return overlap
    
    def filter_regions_overlap(self, regions_overlap: Tensor, 
                               pred_region_class_map: Tensor, target_region_class_map: Tensor):
        """
        Filter the regions overlap tensor to only store the overlap values between regions
        that have the same class label.

        Instead of performing a nested loop to check if the class labels of the regions match,
        we can use the following algorithm:
        - Create a 0-1 mask tensor that has 1s where the class labels of the regions match.
        - Multiply the mask with the regions overlap tensor to get the filtered overlap tensor.
        """
        # Create a 0-1 mask tensor that has 1s where the class labels of the regions match
        class_match_mask = (pred_region_class_map.unsqueeze(0) == target_region_class_map.unsqueeze(1)).float()
        # Multiply the mask with the regions overlap tensor to get the filtered overlap tensor
        filtered_overlap = regions_overlap * class_match_mask
        return filtered_overlap

    def get_rom_rum(self, output, target):
        """
        Calculate the Region-wise over-segmentation measure (ROM) and region-wise under-segmentation measure (RUM).
        """
        pred, target = self.preprocess_inputs(output, target)

        # Get the regions in the predicted and target segmentations
        pred_regions = self.get_regions(pred)
        pred_region_counts = pred_regions.reshape(-1).bincount(minlength=1)
        pred_region_class_map = self.get_region_class_map(pred, pred_regions, pred_region_counts)
        target_regions = self.get_regions(target)
        target_region_counts = target_regions.reshape(-1).bincount(minlength=1)
        target_region_class_map = self.get_region_class_map(target, target_regions, target_region_counts)
        if len(pred_region_counts) <= 1 or len(target_region_counts) <= 1:
            return 1, 1
        
        # Get region overlap values, and for each region in the target segmentation,
        # get the number of regions in the predicted segmentation that overlap with it.
        # Make sure to only consider regions that have the same class label.
        regions_overlap = self.get_regions_overlap(pred_regions, target_regions, 
                                                   pred_region_counts, target_region_counts)
        regions_overlap = self.filter_regions_overlap(regions_overlap, pred_region_class_map, target_region_class_map)
        # Ignore the background label of target_regions
        regions_overlap[:, 0] = 0
        regions_overlap[0, :] = 0

        # Get ROM
        # First get the ground-truth regions that have more than one predicted region overlapping with them.
        target_os = torch.sum(torch.where(regions_overlap >= 1, 1, 0), dim=1)
        # Then get the predicted regions that overlap with any ground-truth region that is over-segmented.
        # NOTE: Thanks to broadcasting, the following multiplication of shapes
        #  (num_target_regions,) and (num_target_regions, num_pred_regions) gives us a valid result.
        pred_os = (target_os > 1).float() @ (regions_overlap >= 1).float()
        # Calculate the ROM while ignoring the background label
        # NOTE: The background label is always at index 0, so we can ignore it by starting from index 1.
        ror = (torch.sum(target_os[1:]) / (len(target_region_counts)-1)) * \
            (torch.sum(pred_os[1:]) / (len(pred_region_counts)-1))
        mo = torch.sum(torch.max(torch.tensor([0]), target_os[1:] - 1))
        rom = math.tanh(ror * mo)

        # Get RUM
        # First get the predicted regions that have more than one ground-truth region overlapping with them.
        pred_us = torch.sum(torch.where(regions_overlap >= 1, 1, 0), dim=0)
        # Then get the ground-truth regions that overlap with any predicted region that is under-segmented.
        target_us = (pred_us > 1).float() @ (regions_overlap.T >= 1).float()
        # Calculate the RUM
        rur = (torch.sum(pred_us[1:]) / (len(pred_region_counts)-1)) * \
            (torch.sum(target_us[1:]) / (len(target_region_counts)-1))
        mu = torch.sum(torch.max(torch.tensor([0]), pred_us[1:] - 1))
        rum = math.tanh(rur * mu)

        return rom, rum

    
if __name__=="__main__":
    # output = Image.open("pred.png")
    # target = Image.open("target.png")
    output: ByteTensor = torch.tensor([
        [0, 0, 1, 1, 2, 2, 255, 255],
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 2, 2, 3, 3],
        [2, 2, 2, 2, 3, 3, 3, 3],
        [2, 2, 2, 2, 3, 3, 3, 3],
        [3, 3, 1, 3, 3, 255, 255, 255],
        [1, 3, 1, 3, 255, 255, 3, 3]
    ]).byte()
    target: ByteTensor = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 2, 2, 3, 3],
        [2, 2, 2, 1, 3, 3, 3, 3],
        [2, 2, 2, 2, 3, 3, 3, 3],
        [3, 3, 3, 3, 255, 255, 255, 255],
        [3, 3, 3, 3, 255, 255, 255, 255]
    ]).byte()

    romrum = ROMRUM()
    print(romrum.get_rom_rum(output, target))