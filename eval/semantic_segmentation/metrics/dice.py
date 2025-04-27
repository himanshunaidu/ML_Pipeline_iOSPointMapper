import numpy as np
import torch

class Dice(object):
    """
    Helps to calculate the Dice coefficient for semantic segmentation, by providing the
    numerator and denominator values for each class.
    Works with both CPU and GPU PyTorch tensors.

    Works with the following assumptions:
    - The input is a PyTorch tensor.
    - The input is a 2D tensor.
    - The input is a ByteTensor, thus the values are between 0 and 255.

    Parameters
    ----------
    num_classes : int
        The number of classes in the dataset.

    epsilon : float
        A small value added to the denominator to avoid division by zero.
    """
    def __init__(self, num_classes=21, epsilon=1e-6, is_output_probabilities=True,
                 min_range=1, max_range=None):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.is_output_probabilities = is_output_probabilities
        self.min_range = min_range
        self.max_range = max_range if max_range is not None else num_classes
    
    def get_dice(self, output, target):
        """
        Calculate the Dice coefficient of a segmentation.

        Parameters
        ----------
        output : torch.ByteTensor
            The output segmentation mask.
            A 4D tensor with dimensions (batch_size, num_classes, height, width).

        target : torch.ByteTensor
            The target segmentation mask.
            A 3D tensor with dimensions (batch_size, height, width).

        Returns
        -------
        area_numerator : numpy.ndarray
            The numerator values for each class.

        area_denominator : numpy.ndarray
            The denominator values for each class.            
        """
        if isinstance(output, tuple):
            output = output[0]

        if self.is_output_probabilities:
            _, pred = torch.max(output, 1)
        else:
            pred = output

        # histc in torch is implemented only for cpu tensors, so move your tensors to CPU
        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()

        pred = pred.type(torch.ByteTensor).clone()
        target = target.type(torch.ByteTensor).clone()

        # shift by 1 so that 255 is 0
        # TODO: Check if this is necessary in the current implementation
        pred += 1
        target += 1

        pred = pred * (target > 0)
        inter = pred * (pred == target)
        area_numerator = torch.histc(inter.float(), bins=self.num_classes, min=self.min_range, max=self.max_range)
        area_pred = torch.histc(pred.float(), bins=self.num_classes, min=self.min_range, max=self.max_range)
        area_mask = torch.histc(target.float(), bins=self.num_classes, min=self.min_range, max=self.max_range)
        area_denominator = area_pred + area_mask + self.epsilon

        return area_numerator.numpy() * 2, area_denominator.numpy()