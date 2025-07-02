import numpy as np
import torch

class F1Score(object):
    """
    Helps to calculate the F1 score metric for semantic segmentation, by providing the precision and recall values
    for each class.
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
        
    is_output_probabilities : bool
        If True, the output is assumed to be probabilities (softmax output).
        If False, the output is assumed to be class indices (argmax output).
    """
    def __init__(self, num_classes=21, epsilon=1e-6, is_output_probabilities=True):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.is_output_probabilities = is_output_probabilities
        
    def get_f1_score(self, output: torch.Tensor, target: torch.Tensor):
        """
        Calculate the F1 score of a segmentation.

        Parameters
        ----------
        output : torch.Tensor
            The output segmentation mask.
            A 4D tensor with dimensions (batch_size, num_classes, height, width) if is_output_probabilities is True,
            or a 3D tensor with dimensions (batch_size, height, width) if is_output_probabilities is False.

        target : torch.Tensor
            The target segmentation mask.
            A 3D tensor with dimensions (batch_size, height, width).

        Returns
        -------
        f1_scores : numpy.ndarray
            The F1 scores for each class.
            
        precisions : numpy.ndarray
            The precision values for each class.
            
        recalls : numpy.ndarray
            The recall values for each class.
        """
        if isinstance(output, tuple):
            output = output[0]

        if self.is_output_probabilities:
            _, pred = torch.max(output, 1)
        else:
            pred = output

        f1_scores = np.zeros(self.num_classes)
        precisions = np.zeros(self.num_classes)
        recalls = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            true_positive = ((pred == i) & (target == i)).sum().item()
            false_positive = ((pred == i) & (target != i)).sum().item()
            false_negative = ((pred != i) & (target == i)).sum().item()

            precision = true_positive / (true_positive + false_positive + self.epsilon)
            recall = true_positive / (true_positive + false_negative + self.epsilon)

            f1_scores[i] = 2 * (precision * recall) / (precision + recall + self.epsilon)
            precisions[i] = precision
            recalls[i] = recall

        return f1_scores, precisions, recalls