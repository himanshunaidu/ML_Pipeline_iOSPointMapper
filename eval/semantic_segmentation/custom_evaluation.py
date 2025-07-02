"""
This script contains a custom evaluation function for evaluating the results of a semantic segmentation experiment.
"""
import torch
import time

from eval.utils import AverageMeter
from eval.semantic_segmentation.metrics.iou import IOU
from eval.semantic_segmentation.metrics.dice import Dice
from eval.semantic_segmentation.metrics.f1 import F1Score
from eval.semantic_segmentation.metrics.persello import Persello
from eval.semantic_segmentation.metrics.rom_rum import ROMRUM
from eval.semantic_segmentation.metrics.old.persello import segmentation_score_Persello as Persello_old, cityscapesIdToClassMap
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
                 idToClassMap=cityscapesIdToClassMap,
                 miou_min_range=1, miou_max_range=None,
                 args):
        """
        Initialize the CustomEvaluation class.
        """
        # Save the arguments
        self.args = args
        self.is_output_probabilities = is_output_probabilities
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.idToClassMap = idToClassMap
        self.miou_min_range = miou_min_range
        self.miou_max_range = miou_max_range if miou_max_range is not None else num_classes

        self.losses = AverageMeter()
        self.batch_time = AverageMeter()
        self.inter_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.f1_score_meter = AverageMeter()
        self.precision_meter = AverageMeter()
        self.recall_meter = AverageMeter()
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

        self.miou_class = IOU(num_classes=self.num_classes, 
                              is_output_probabilities=self.is_output_probabilities,
                              min_range=self.miou_min_range, max_range=self.miou_max_range)
        self.dice_class = Dice(num_classes=self.num_classes, 
                               is_output_probabilities=self.is_output_probabilities,
                               min_range=self.miou_min_range, max_range=self.miou_max_range)
        self.f1_score_class = F1Score(num_classes=self.num_classes,
                                      is_output_probabilities=self.is_output_probabilities)
        self.persello_class = Persello(num_classes=self.num_classes, max_regions=self.max_regions, 
                                       is_output_probabilities=self.is_output_probabilities)
        self.romrum_class = ROMRUM(num_classes=self.num_classes, max_regions=self.max_regions,
                                   is_output_probabilities=self.is_output_probabilities)

        # To record time taken for each metric
        self.miou_times = []
        self.dice_times = []
        self.f1_score_times = []
        self.persello_times = []
        self.romrum_times = []
        self.persello_old_times = []
        self.romrum_old_times = []

    def preprocess_for_old_metrics(self, output: torch.Tensor, target: torch.Tensor) -> tuple:
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

        Returns
        -------
        tuple
            A tuple containing the preprocessed output and target numpy arrays.
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
        
        pred: torch.ByteTensor = pred.type(torch.ByteTensor).clone()
        target: torch.ByteTensor = target.type(torch.ByteTensor).clone()

        pred = pred.numpy()[0]
        target = target.numpy()[0]

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
        # Clone the output and target tensors to avoid modifying the original tensors
        # if isinstance(output, tuple):
        #     output = output[0]
        # output = output.clone()
        # target = target.clone()

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
        
        # Calculate F1 Score, Precision, and Recall
        f1_start_time = time.time()
        f1_scores, precisions, recalls = self.f1_score_class.get_f1_score(output, target)
        self.f1_score_times.append(time.time() - f1_start_time)
        self.f1_score_meter.update(f1_scores)
        self.precision_meter.update(precisions)
        self.recall_meter.update(recalls)

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
        pred_pre_old, target_pre_old = self.preprocess_for_old_metrics(output, target)

        # Calculate Persello (Old)
        persello_old_start_time = time.time()
        persello_old_over, persello_old_under = 0, 0
        for class_id in self.idToClassMap.keys():
            persello_old_over_c, persello_old_under_c = Persello_old(target_pre_old, pred_pre_old, class_id)
            persello_old_over += persello_old_over_c
            persello_old_under += persello_old_under_c
        self.persello_old_over_list.append(persello_old_over)
        self.persello_old_under_list.append(persello_old_under)
        self.persello_old_over_meter.update(persello_old_over/len(self.idToClassMap.keys()))
        self.persello_old_under_meter.update(persello_old_under/len(self.idToClassMap.keys()))
        self.persello_old_times.append(time.time() - persello_old_start_time)

        # Calculate ROM/RUM (Old)
        romrum_old_start_time = time.time()
        romrum_old_over, romrum_old_under = 0, 0
        for class_id in self.idToClassMap.keys():
            romrum_old_over_c, romrum_old_under_c = ROMRUM_old(target_pre_old, pred_pre_old, class_id)
            romrum_old_over += romrum_old_over_c
            romrum_old_under += romrum_old_under_c
        self.romrum_old_over_list.append(romrum_old_over)
        self.romrum_old_under_list.append(romrum_old_under)
        self.romrum_old_over_meter.update(romrum_old_over/len(self.idToClassMap.keys()))
        self.romrum_old_under_meter.update(romrum_old_under/len(self.idToClassMap.keys()))
        self.romrum_old_times.append(time.time() - romrum_old_start_time)

    def get_results(self):
        """
        Get the evaluation results.

        Returns
        -------
        dict
            A dictionary containing the evaluation results.
        """
        iou = self.inter_meter.sum / (self.union_meter.sum + 1e-10)
        dice = self.dice_numerator_meter.sum / (self.dice_denominator_meter.sum + 1e-10)
        miou = iou.mean().item() # Need to item() to convert from array to scalar
        mdice = dice.mean().item() # Need to item() to convert from array to scalar
        
        f1_score = self.f1_score_meter.sum / (self.f1_score_meter.count + 1e-10)
        precision = self.precision_meter.sum / (self.precision_meter.count + 1e-10)
        recall = self.recall_meter.sum / (self.recall_meter.count + 1e-10)
        m_f1_score = f1_score.mean().item()  # Need to item() to convert from array to scalar
        m_precision = precision.mean().item()  # Need to item() to convert from array to scalar
        m_recall = recall.mean().item()  # Need to item() to convert from array to scalar

        persello_over = self.persello_over_meter.sum / (self.persello_over_meter.count + 1e-10)
        persello_under = self.persello_under_meter.sum / (self.persello_under_meter.count + 1e-10)
        romrum_over = self.romrum_over_meter.sum / (self.romrum_over_meter.count + 1e-10)
        romrum_under = self.romrum_under_meter.sum / (self.romrum_under_meter.count + 1e-10)

        persello_old_over = self.persello_old_over_meter.sum / (self.persello_old_over_meter.count + 1e-10)
        persello_old_under = self.persello_old_under_meter.sum / (self.persello_old_under_meter.count + 1e-10)
        romrum_old_over = self.romrum_old_over_meter.sum / (self.romrum_old_over_meter.count + 1e-10)
        romrum_old_under = self.romrum_old_under_meter.sum / (self.romrum_old_under_meter.count + 1e-10)

        results = {
            'mIoU': miou,
            'mDice': mdice,
            'mF1_Score': m_f1_score,
            'mPrecision': m_precision,
            'mRecall': m_recall,
            'Persello_Oversegmentation': persello_over,
            'Persello_Undersegmentation': persello_under,
            'ROM_RUM_Oversegmentation': romrum_over,
            'ROM_RUM_Undersegmentation': romrum_under,
            # 'Persello_Oversegmentation_Old': persello_old_over,
            # 'Persello_Undersegmentation_Old': persello_old_under,
            'ROM_RUM_Oversegmentation_Old': romrum_old_over,
            'ROM_RUM_Undersegmentation_Old': romrum_old_under,

            'mIoU_times': self.miou_times,
            'mDice_times': self.dice_times,
            'mF1_Score_times': self.f1_score_times,
            'Persello_times': self.persello_times,
            'ROM_RUM_times': self.romrum_times,
            'Persello_Old_times': self.persello_old_times,
            'ROM_RUM_Old_times': self.romrum_old_times,
        }
        return results

