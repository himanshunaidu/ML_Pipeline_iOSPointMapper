import numpy as np
import torch
from torch import Tensor, ByteTensor
from skimage.measure import label
import time

class Persello(object):
    """
    Helps to calculate the Persello metric for semantic segmentation.

    Assumes that there are at maximum 255 regions in the segmentation.

    TODO: Check if the error metric has to take the segmentation labels into account.
    Currently, the error is only region-based, and not label-based.
    """
    def __init__(self, num_classes=21, epsilon=1e-6):
        self.num_classes = num_classes
        self.epsilon = epsilon

    def preprocess_inputs(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        _, pred = torch.max(output, 1)

        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()
        
        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)
        
        # shift by 1 so that 255 is 0
        pred += 1
        target += 1

        return pred, target

    def get_regions(self, img: Tensor):
        """
        Get the number of regions in an image using skimage.
        # NOTE: Follows the assumption that there are at maximum 255 regions in the segmentation
        """
        # detect contiguous regions
        img_numpy = img.numpy()
        img_numpy = img_numpy.astype(np.uint8)
        regions = label(img_numpy)
        # Truncate the regions to 255
        regions[regions > 255] = 255
        return torch.from_numpy(regions)

    def get_regions_overlap(self, pred_regions: Tensor, target_regions: Tensor,
                            pred_region_bin_counts: Tensor, target_region_bin_counts: Tensor):
        """
        Get the overlap between regions in the predicted and target segmentations.
        
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
        """
        pred, target = self.preprocess_inputs(output, target)

        # Get the regions in the predicted and target segmentations
        pred_regions = self.get_regions(pred)
        pred_region_counts = pred_regions.reshape(-1).bincount(minlength=1)
        target_regions = self.get_regions(target)
        target_region_counts = target_regions.reshape(-1).bincount(minlength=1)
        if len(pred_region_counts) <= 1 or len(target_region_counts) <= 1:
            return 1, 1

        # Get region overlap values, and for each region in the target segmentation,
        # get the region in the predicted segmentation that has the maximum overlap with it.
        # While doing the region matches, avoid the background label of pred_regions.
        regions_overlap = self.get_regions_overlap(pred_regions, target_regions, 
                                                   pred_region_counts, target_region_counts)
        region_matches = torch.argmax(regions_overlap[:, 1:], dim=1)+1

        # Calculate the Persello metrics
        oversegmentation_errors = self.over_segmentation_error(
            regions_overlap, region_matches, pred_region_counts, target_region_counts
        )
        undersegmentation_errors = self.under_segmentation_error(
            regions_overlap, region_matches, pred_region_counts, target_region_counts
        )
        # Ignore the background label of target_regions
        return oversegmentation_errors[1:].mean().item(), undersegmentation_errors[1:].mean().item()

if __name__=="__main__":
    output: ByteTensor = torch.tensor(np.random.randint(0, 255, (1, 21, 256, 256))).byte()
    target: ByteTensor = torch.tensor(np.random.randint(0, 255, (256, 256))).byte()
    output = ByteTensor(np.array(output))
    target = ByteTensor(np.array(target))