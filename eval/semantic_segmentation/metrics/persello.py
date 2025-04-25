import numpy as np
import torch
from torch import Tensor, ByteTensor
from skimage.measure import label
import time

class Persello(object):
    """
    Helps to calculate the Persello metric for semantic segmentation.

    NOTE: While the function takes in 4D tensors, the first dimension, is supposed to be only 1 for the time being.
    """
    def __init__(self, num_classes=21, epsilon=1e-6, max_regions=255, is_output_probabilities=True):
        self.num_classes = num_classes
        self.epsilon = epsilon
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

    def get_regions(self, img: Tensor, background_label=0):
        """
        Get the number of regions in an image using skimage.
        # NOTE: Follows the assumption that there are at maximum 255 regions in the segmentation
        """
        # detect contiguous regions
        img_numpy = img.numpy()
        img_numpy = img_numpy.astype(np.uint8)
        regions = label(img_numpy, background=background_label)
        # Truncate the regions to 255
        regions[regions > self.max_regions] = self.max_regions
        return torch.from_numpy(regions)

    def get_region_class_map(self, img: Tensor, regions: Tensor, region_bin_counts: Tensor):
        """
        Get the mapping of region labels to class labels.
        Helps get the class to which the region belongs to.
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


    def over_segmentation_error(self, regions_overlap: Tensor, region_matches: Tensor,
                                pred_region_bin_counts: Tensor, target_region_bin_counts: Tensor):
        """
        Calculate the over-segmentation error of a segmentation.

        For each region in the target segmentation, the over-segmentation error is calculated as:
        1 - (overlap / target_mod)
        """
        oversegmentation_errors = 1 - (
            regions_overlap[torch.arange(len(region_matches)), region_matches] / \
                (target_region_bin_counts+self.epsilon)
        )
        return oversegmentation_errors
        

    def under_segmentation_error(self, regions_overlap: Tensor, region_matches: Tensor,
                                pred_region_bin_counts: Tensor, target_region_bin_counts: Tensor):
        """
        Calculate the under-segmentation error of a segmentation.
        """
        undersegmentation_errors = 1 - (
            regions_overlap[torch.arange(len(region_matches)), region_matches] / \
                (pred_region_bin_counts[region_matches]+self.epsilon)
        )
        return undersegmentation_errors

    def get_persello(self, output, target):
        """
        Calculate the Persello metric for semantic segmentation.

        Algorithm:
        - Get the regions in the predicted and target segmentations.
        - Get the region class map for each segmentation.
        - Get the overlap between regions in the predicted and target segmentations.
        - Filter the regions overlap tensor to only store the overlap values between regions
          that have the same class label.
        - Calculate the over-segmentation error for each region in the target segmentation.
        - Calculate the under-segmentation error for each region in the target segmentation.
        - Return the mean over-segmentation and under-segmentation errors.
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
        # get the region in the predicted segmentation that has the maximum overlap with it.
        # Make sure to only consider regions that have the same class label.
        regions_overlap = self.get_regions_overlap(pred_regions, target_regions, 
                                                   pred_region_counts, target_region_counts)
        regions_overlap = self.filter_regions_overlap(regions_overlap, pred_region_class_map, target_region_class_map)
        # Ignore the background label of target_regions
        regions_overlap[:, 0] = 0
        regions_overlap[0, :] = 0
        region_matches = torch.argmax(regions_overlap, dim=1)

        # Calculate the Persello metrics
        oversegmentation_errors = self.over_segmentation_error(
            regions_overlap, region_matches, pred_region_counts, target_region_counts
        )[1:]
        undersegmentation_errors = self.under_segmentation_error(
            regions_overlap, region_matches, pred_region_counts, target_region_counts
        )[1:]
        # Ignore the background label of target_regions
        # print(oversegmentation_errors, undersegmentation_errors)
        return oversegmentation_errors.mean().item(), undersegmentation_errors.mean().item()

if __name__=="__main__":
    output: ByteTensor = torch.tensor(np.random.randint(0, 3, (1, 4, 4, 4))).byte()
    target: ByteTensor = torch.tensor([[2, 2, 2, 0],
        [255, 2, 1, 0],
        [255, 2, 1, 1],
        [255, 2, 2, 1]]).byte()
    # output: ByteTensor = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4]).byte()
    # target: ByteTensor = torch.tensor([0, 1, 2, 2, 3, 3, 4, 4]).byte()

    persello = Persello()
    error = persello.get_persello(output, target)
    print(error)