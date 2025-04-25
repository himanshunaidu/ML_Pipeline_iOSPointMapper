"""
This script contains a custom evaluation function for evaluating the results of a semantic segmentation experiment.
"""
import torch
import time

from eval.utils import AverageMeter
from eval.semantic_segmentation.metrics.iou import IOU
from eval.semantic_segmentation.metrics.dice import Dice
from eval.semantic_segmentation.metrics.persello import Persello
from eval.semantic_segmentation.metrics.rom_rum import ROMRUM
from eval.semantic_segmentation.metrics.old.persello import segmentation_score_Persello as Persello_old, idToClassMap
from eval.semantic_segmentation.metrics.old.rom_rum import rom_rum as ROMRUM_old

class CustomEvaluation:
    """
    Custom evaluation class for evaluating the results of a semantic segmentation experiment.

    Metrics to be calculated:
    - Mean Intersection over Union (mIoU)
    - Mean Dice Coefficient (mDice)
    - Persello Scores (Oversegmentation and Undersegmentation), both old version and new version
    - ROM/RUM Scores (Region-wise Oversegmentation and Region-wise Undersegmentation Measures)

    Also calculates the time required to calculate the metrics.
    """
    def __init__(self, *, num_classes, max_regions=1024, is_output_probabilities=True,
                 args):
        """
        Initialize the CustomEvaluation class.
        """
        self.losses = AverageMeter()
        self.batch_time = AverageMeter()
        self.inter_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.dice_numerator_meter = AverageMeter()
        self.dice_denominator_meter = AverageMeter()
        self.persello_over_list = []
        self.persello_under_list = []
        self.persello_over_meter = AverageMeter()
        self.persello_under_meter = AverageMeter()
        self.romrum_over_list = []
        self.romrum_under_list = []
        self.romrum_over_meter = AverageMeter()
        self.romrum_under_meter = AverageMeter()

        self.persello_old_over_list = []
        self.persello_old_under_list = []
        self.persello_old_over_meter = AverageMeter()
        self.persello_old_under_meter = AverageMeter()
        self.romrum_old_over_list = []
        self.romrum_old_under_list = []
        self.romrum_old_over_meter = AverageMeter()
        self.romrum_old_under_meter = AverageMeter()

        self.miou_class = IOU(num_classes=num_classes, 
                              is_output_probabilities=is_output_probabilities)
        self.dice_class = Dice(num_classes=num_classes, 
                               is_output_probabilities=is_output_probabilities)
        self.persello_class = Persello(num_classes=num_classes, max_regions=max_regions, 
                                       is_output_probabilities=is_output_probabilities)
        self.romrum_class = ROMRUM(num_classes=num_classes, max_regions=max_regions,
                                   is_output_probabilities=is_output_probabilities)

        # To record time taken for each metric
        self.miou_times = []
        self.dice_times = []
        self.persello_times = []
        self.romrum_times = []
        self.persello_old_times = []
        self.romrum_old_times = []

        # Save the arguments
        self.args = args
        self.is_output_probabilities = is_output_probabilities
        self.num_classes = num_classes
        self.max_regions = max_regions

    def preprocess_for_old_metrics(self, output: torch.Tensor, target: torch.Tensor):
        """
        Preprocess the output and target tensors for the old metrics.

        Parameters
        ----------
        output : torch.Tensor
            The output segmentation mask.
            A 4D tensor with dimensions (batch_size, num_classes, height, width).

        target : torch.Tensor
            The target segmentation mask.
            A 3D tensor with dimensions (batch_size, height, width).
        """
        if isinstance(output, tuple):
            output = output[0]

        if self.is_output_probabilities:
            _, pred = torch.max(output, 1)
        else:
            pred = output.clone()
        
        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()
        
        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        return pred, target

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        Update the evaluation metrics with the given output and target.

        Parameters
        ----------
        output : torch.Tensor
            The output segmentation mask.
            A 4D tensor with dimensions (batch_size, num_classes, height, width).

        target : torch.Tensor
            The target segmentation mask.
            A 3D tensor with dimensions (batch_size, height, width).
        """
        # Calculate mIoU
        iou_start_time = time.time()
        area_inter, area_union = self.miou_class.get_iou(output, target)
        self.miou_times.append(time.time() - iou_start_time)
        self.inter_meter.update(area_inter)
        self.union_meter.update(area_union)

        # Calculate mDice
        dice_start_time = time.time()
        area_numerator, area_denominator = self.dice_class.get_dice(output, target)
        self.dice_times.append(time.time() - dice_start_time)
        self.dice_numerator_meter.update(area_numerator)
        self.dice_denominator_meter.update(area_denominator)

        # Calculate Persello (New)
        persello_start_time = time.time()
        persello_over, persello_under = self.persello_class.get_persello(output, target)
        self.persello_times.append(time.time() - persello_start_time)
        self.persello_over_list.append(persello_over)
        self.persello_under_list.append(persello_under)
        self.persello_over_meter.update(persello_over)
        self.persello_under_meter.update(persello_under)

        # Calculate ROM/RUM (New)
        romrum_start_time = time.time()
        romrum_over, romrum_under = self.romrum_class.get_rom_rum(output, target)
        self.romrum_times.append(time.time() - romrum_start_time)
        self.romrum_over_list.append(romrum_over)
        self.romrum_under_list.append(romrum_under)
        self.romrum_over_meter.update(romrum_over)
        self.romrum_under_meter.update(romrum_under)

        ## Before calculating the old versions, we need to convert the output and target tensors
        # to numpy arrays and map the class indices to the old class indices

