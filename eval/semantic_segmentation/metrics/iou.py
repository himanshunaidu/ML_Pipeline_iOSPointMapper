import numpy as np
import torch

class IOU(object):
    """
    Helps to calculate the intersection over union (mIoU) metric for semantic segmentation, by providing the
    intersection and union values for each class.
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
        A small value added to the union to avoid division by zero.
    """
    def __init__(self, num_classes=21, epsilon=1e-6, is_output_probabilities=True):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.is_output_probabilities = is_output_probabilities

    def get_iou(self, output, target):
        """
        Calculate the intersection over union coefficient of a segmentation.

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
        area_inter : numpy.ndarray
            The intersection values for each class.

        area_union : numpy.ndarray
            The union values for each class.
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

        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        # shift by 1 so that 255 is 0
        # TODO: Check if this is necessary in the current implementation
        pred += 1
        target += 1

        pred = pred * (target > 0)
        inter = pred * (pred == target)
        area_inter = torch.histc(inter.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_pred = torch.histc(pred.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_mask = torch.histc(target.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_union = area_pred + area_mask - area_inter + self.epsilon

        return area_inter.numpy(), area_union.numpy()