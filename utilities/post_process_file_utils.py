"""
This utility file contains functions used by the adhoc test scripts for various post-processing purposes.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import torch
from PIL import Image
from tqdm import tqdm
import glob
import re

from torch import nn
from torchvision.transforms import functional as F
from torch.utils import data

from utilities.print_utils import *

class ResultsTest:
    def __init__(self, path_stem: str, target_classes: dict, target_datasets: dict[str, list[str]], *, 
                 model_name: str = "Unknown", output_dir: Optional[str] = "results_test/processed_results"):
        self.path_stem = path_stem
        self.target_classes = target_classes
        # The target datasets in the results_test directory can be a combination of different datasets.
        self.target_datasets = target_datasets
        self.model_name = model_name
        self.output_dir = output_dir

        self.target_dataset_result_paths = self.get_result_paths()
    
    def get_result_paths(self) -> dict[str, List[str]]:
        """
        Get the result paths for each dataset based on the path stem and target datasets.
        Returns a dictionary where keys are dataset names and values are lists of paths.
        """
        dataset_results_paths: dict[str, list[str]] = {dataset: [] for dataset in self.target_datasets.keys()}
        for path in glob.glob(self.path_stem):
            if not os.path.exists(path):
                continue
            dataset_name = re.search(r'results_test/[^/]*/(.*?)/', path).group(1)
            for dataset, dataset_path in self.target_datasets.items():
                if dataset_name in dataset_path:
                    dataset_results_paths[dataset].append(path)
        return dataset_results_paths

common_target_classes = {
    "sidewalk": 1,
    "building": 2,
    "traffic_sign": 5,
    "traffic_light": 4,
    "pole": 3
}
common_datasets = {
    "Cityscapes": ["results_city_val"],
    "OASIS": ["results_edge_mapping_ios_val"],
    "iOS": ["results_ios_point_mapper_val"],
    "Combined": ["results_edge_mapping_ios_val", "results_ios_point_mapper_val"]
}

def get_metrics_table(results: dict) -> pd.DataFrame:
    """
    Loads class metrics from results and returns a DataFrame.
    
    Args:
        results (dict): Dictionary containing the results, expected to have a 'class_metrics' key.
    
    Returns:
        pd.DataFrame: Table of per-class metrics
    """
    # Load if path is given

    # Extract class-level metrics
    class_metrics = results.get("class_metrics", [])
    if not class_metrics:
        raise ValueError("No 'class_metrics' key found in results.")

    # Create a DataFrame
    df = pd.DataFrame(class_metrics)

    # Optional: Sort by class_id or name
    df.sort_values(by="class_id", inplace=True)

    # Optional: Select and rename columns for cleaner output
    df = df[[
        "class_id",
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "iou_score",
        "pixel_count"
    ]].rename(columns={
        "class_id": "Class ID",
        "class_name": "Class",
        "iou_score": "IoU",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
        "pixel_count": "Pixel Count"
    })

    return df

def get_post_metrics(args: ResultsTest) -> dict:
    target_classes = args.target_classes
    if target_classes is None:
        print_error_message('Target classes are not provided in the config. Please check the config file.')
        return
    
    class_stats = []
    total_pixels = 0
    weighted_iou_sum = 0.0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    all_iou_values = []
    all_precision_values = []
    all_recall_values = []
    
    tp, fp, tn, fn = 0, 0, 0, 0
    precision, recall, specificity, f1_score, iou_score = 0.0, 0.0, 0.0, 0.0, 0.0
    
    datasets = args.target_dataset_result_paths
    for index, target_class in enumerate(target_classes.items()):
        target_class_name = target_class[0]
        target_class_id = target_class[1]

        tp, fp, tn, fn = 0, 0, 0, 0
        precision, recall, specificity, f1_score, iou_score = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for dataset_name, dataset_paths in datasets.items():
            for dataset_path in dataset_paths:
                input_dir = os.path.join(dataset_path, 'input')
                pred_dir = os.path.join(dataset_path, 'pred')
                target_dir = os.path.join(dataset_path, 'target')
                
                for pred_file_name in tqdm(os.listdir(pred_dir), desc=f"Processing class {target_class_id}"):
                    if not pred_file_name.endswith('.png'):
                        continue
                    target_file_name = pred_file_name.replace('pred', 'target')
                    
                    # Load predicted mask
                    pred_mask = np.array(Image.open(os.path.join(pred_dir, pred_file_name)))  # shape: [H, W]
                    # Load corresponding ground truth mask
                    gt_mask = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]
                    
                    # Create a mask that only contains the target class
                    pred_mask_target = (pred_mask == target_class_id).astype(np.uint8)
                    gt_mask_target = (gt_mask == target_class_id).astype(np.uint8)
                    
                    # NOTE: Skip if ground truth mask is empty
                    if np.sum(gt_mask_target) == 0: continue
                    
                    # Calculate True Positives, False Positives, True Negatives, False Negatives
                    tp += np.sum((pred_mask_target == 1) & (gt_mask_target == 1))
                    fp += np.sum((pred_mask_target == 1) & (gt_mask_target == 0))
                    tn += np.sum((pred_mask_target == 0) & (gt_mask_target == 0))
                    fn += np.sum((pred_mask_target == 0) & (gt_mask_target == 1))
        
        # Calculate metrics
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        # print_info_message(f"precision: {precision}, recall: {recall}, specificity: {specificity}")
        f1_score = 2 * (precision * recall) / (precision + recall) if (tp + fn > 0 and tp + fp > 0) else 0.0
        iou_score = tp / (tp + fp + fn) if (tp + fp + fn > 0) else 0.0
        
        pixel_count = tp + fn # Total pixels in the target class
        
        class_stats.append({
            'class_id': target_class,
            'class_name': target_class_name,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'iou_score': iou_score,
            'pixel_count': pixel_count
        })
        
        total_pixels += pixel_count
        weighted_iou_sum += iou_score * pixel_count
        all_iou_values.append(iou_score)
        weighted_precision_sum += precision * pixel_count
        all_precision_values.append(precision)
        weighted_recall_sum += recall * pixel_count
        all_recall_values.append(recall)
        
    # Macro (mean) metrics
    mIoU = np.mean(all_iou_values) if all_iou_values else 0.0
    FWIoU = weighted_iou_sum / total_pixels if total_pixels > 0 else 0.0
    mPrecision = np.mean(all_precision_values) if all_precision_values else 0.0
    FWmPrecision = weighted_precision_sum / total_pixels if total_pixels > 0 else 0.0
    mRecall = np.mean(all_recall_values) if all_recall_values else 0.0
    FWmRecall = weighted_recall_sum / total_pixels if total_pixels > 0 else 0.0
    
    results = {
        'dataset': args.dataset_name,
        'model': args.model_name,
        'num_classes': len(args.target_classes),
        'eval_classes': list(args.target_classes.keys()),
        'eval_class_names': list(args.target_classes.values()),
        'class_metrics': class_stats,
        'overall_metrics': {
            'mIoU': mIoU,
            'FWIoU': FWIoU,
            'mPrecision': mPrecision,
            'FWmPrecision': FWmPrecision,
            'mRecall': mRecall,
            'FWmRecall': FWmRecall,
        }
    }
    
    return results

def get_post_metrics(args: ResultsTest) -> dict:
    target_classes = args.target_classes
    if target_classes is None:
        print_error_message('Target classes are not provided in the config. Please check the config file.')
        return
    
    post_metrics = {
        'model': args.model_name,
    }
    
    datasets = args.target_dataset_result_paths      
    # For each dataset (group of results), for each target class, we parse through the results of all sub-datasets and calculate the metrics.  
    for dataset_name, dataset_paths in datasets.items():
        print(f"Processing dataset: {dataset_name} with {len(dataset_paths)} sub-datasets")
        class_stats = []
        total_pixels = 0
        weighted_iou_sum = 0.0
        weighted_precision_sum = 0.0
        weighted_recall_sum = 0.0
        all_iou_values = []
        all_precision_values = []
        all_recall_values = []
        
        for index, target_class in enumerate(target_classes.items()):
            print(f"Processing class {index + 1}/{len(target_classes)}: {target_class}")
            target_class_name = target_class[0]
            target_class_id = target_class[1]
            
            tp, fp, tn, fn = 0, 0, 0, 0
            precision, recall, specificity, f1_score, iou_score = 0.0, 0.0, 0.0, 0.0, 0.0

            for index, dataset_path in enumerate(dataset_paths):
                input_dir = os.path.join(dataset_path, 'input')
                pred_dir = os.path.join(dataset_path, 'pred')
                target_dir = os.path.join(dataset_path, 'target')
                
                for pred_file_name in tqdm(os.listdir(pred_dir), desc=f"Processing class {target_class_id} in sub-dataset {index + 1}/{len(dataset_paths)}"):
                    if not pred_file_name.endswith('.png'):
                        continue
                    target_file_name = pred_file_name.replace('pred', 'target')
                    
                    # Load predicted mask
                    pred_mask = np.array(Image.open(os.path.join(pred_dir, pred_file_name)))  # shape: [H, W]
                    # Load corresponding ground truth mask
                    gt_mask = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]
                    
                    # Create a mask that only contains the target class
                    pred_mask_target = (pred_mask == target_class_id).astype(np.uint8)
                    gt_mask_target = (gt_mask == target_class_id).astype(np.uint8)
                    
                    # NOTE: Skip if ground truth mask is empty
                    if np.sum(gt_mask_target) == 0: continue
                    
                    # Calculate True Positives, False Positives, True Negatives, False Negatives
                    tp += np.sum((pred_mask_target == 1) & (gt_mask_target == 1))
                    fp += np.sum((pred_mask_target == 1) & (gt_mask_target == 0))
                    tn += np.sum((pred_mask_target == 0) & (gt_mask_target == 0))
                    fn += np.sum((pred_mask_target == 0) & (gt_mask_target == 1))
        
            # Calculate metrics
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
            # print_info_message(f"precision: {precision}, recall: {recall}, specificity: {specificity}")
            f1_score = 2 * (precision * recall) / (precision + recall) if (tp + fn > 0 and tp + fp > 0) else 0.0
            iou_score = tp / (tp + fp + fn) if (tp + fp + fn > 0) else 0.0
            
            pixel_count = tp + fn # Total pixels in the target class
            
            class_stats.append({
                'class_id': target_class,
                'class_name': target_class_name,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'iou_score': iou_score,
                'pixel_count': pixel_count
            })

            total_pixels += pixel_count
            weighted_iou_sum += iou_score * pixel_count
            all_iou_values.append(iou_score)
            weighted_precision_sum += precision * pixel_count
            all_precision_values.append(precision)
            weighted_recall_sum += recall * pixel_count
            all_recall_values.append(recall)
        
        # Macro (mean) metrics
        mIoU = np.mean(all_iou_values) if all_iou_values else 0.0
        FWIoU = weighted_iou_sum / total_pixels if total_pixels > 0 else 0.0
        mPrecision = np.mean(all_precision_values) if all_precision_values else 0.0
        FWmPrecision = weighted_precision_sum / total_pixels if total_pixels > 0 else 0.0
        mRecall = np.mean(all_recall_values) if all_recall_values else 0.0
        FWmRecall = weighted_recall_sum / total_pixels if total_pixels > 0 else 0.0
        
        results = {
            'num_classes': len(args.target_classes),
            'eval_classes': list(args.target_classes.keys()),
            'eval_class_names': list(args.target_classes.values()),
            'class_metrics': class_stats,
            'overall_metrics': {
                'mIoU': mIoU,
                'FWIoU': FWIoU,
                'mPrecision': mPrecision,
                'FWmPrecision': FWmPrecision,
                'mRecall': mRecall,
                'FWmRecall': FWmRecall,
            }
        }
        post_metrics[dataset_name] = results

    return post_metrics