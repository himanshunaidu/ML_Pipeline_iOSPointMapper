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

import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score

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

# Temp code:
ios_point_mapper_to_cocoStuff_custom_11_dict = {0:10, 1:9, 2:255, 3:255, 4:2, 5:9, 6:9, 7:0, 8:1, 9:1, 10:9, 
    11:8, 12:8, 13:8, 14:9, 15:9, 16:8, 17:9, 18:3, 19:8, 20:9,
    21:0, 22:1, 23:8, 24:8, 25:1, 26:7, 27:4, 28:5, 29:9, 30:9,
    31:255, 32:6, 33:8}

cityscapes_cmap = np.zeros((256, 3), dtype=np.uint8)
cityscapes_cmap[0] = [128, 64, 128]  # Road
cityscapes_cmap[1] = [244, 35, 232]  # Sidewalk
cityscapes_cmap[2] = [70, 70, 70]   # Building
cityscapes_cmap[5] = [153, 153, 153]  # Pole
cityscapes_cmap[6] = [250, 170, 30]  # Traffic Light
cityscapes_cmap[7] = [220, 220, 0]  # Traffic Sign
cityscapes_cmap[8] = [107, 142, 35]  # Vegetation
cityscapes_cmap[9] = [152, 251, 152]  # Terrain

common_cmap_base = np.array([
    [128, 64, 128], # Road
    [244, 35,232], # Sidewalk
    [70, 70, 70],  # Building
    [153,153,153], # Pole
    [250,170, 30], # Traffic Light
    [220,220,  0], # Traffic Sign
    [107,142, 35], # Vegetation
    [152,251,152], # Terrain
], dtype=np.uint8)
common_cmap = np.zeros((256, 3), dtype=np.uint8)
common_cmap[:len(common_cmap_base)] = common_cmap_base

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
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall > 0.0) else 0.0
            iou_score = tp / (tp + fp + fn) if (tp + fp + fn > 0) else 0.0
            
            pixel_count = tp + fn # Total pixels in the target class
            
            class_stats.append({
                'class_id': target_class_id,
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

def create_csv_from_metrics(metrics: dict, output_path: str) -> pd.DataFrame:
    """
    Creates a CSV file from the metrics dictionary.
    
    Args:
        metrics (dict): Dictionary containing the metrics.
        output_path (str): Path to save the CSV file.
    """
    CSV_COLUMNS = [
        "model", "dataset", "class_id", "class_name",
        "iou_score", "precision", "recall", "f1_score", "pixel_count"
    ]
    
    for dataset_name, dataset_metrics in metrics.items():
        if not isinstance(dataset_metrics, dict): continue
        df = pd.DataFrame(columns=CSV_COLUMNS)
        
        for class_metric in dataset_metrics['class_metrics']:
            row = {
                "model": metrics['model'],
                "dataset": dataset_name,
                "class_id": class_metric['class_id'],
                "class_name": str.capitalize(class_metric['class_name']),
                "iou_score": f"{class_metric['iou_score']:.4f}",
                "precision": f"{class_metric['precision']:.4f}",
                "recall": f"{class_metric['recall']:.4f}",
                "f1_score": f"{class_metric['f1_score']:.4f}",
                "pixel_count": class_metric['pixel_count']
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    return df


def get_radar_graph_for_classes(model_results: dict[str, dict], target_classes: dict[str, int],
                    target_dataset: tuple[str, list[str]], radar_metric: str = "iou_score") -> go.Figure:
    """
    Generates a Plotly radar graph of multiple models from the metrics.
    """
    fig = go.Figure()
    
    categories = list(target_classes.keys())
    
    for model_name, metrics in model_results.items():
        if target_dataset[0] not in metrics:
            print_error_message(f"Dataset {target_dataset[0]} not found in model results for {model_name}.")
            continue
        
        dataset_metrics = metrics[target_dataset[0]]
        class_metrics = dataset_metrics['class_metrics']
        
        # Extract the metric values for the target classes
        metric_values = []
        for class_name in categories:
            class_id = target_classes[class_name]
            class_metric = next((cm for cm in class_metrics if cm['class_name'] == class_name), None)
            if class_metric:
                metric_values.append(class_metric[radar_metric])
            else:
                metric_values.append(0.0)
        
        fig.add_trace(go.Scatterpolar(
            r=metric_values + [metric_values[0]],  # Close the loop
            theta=categories + [categories[0]],  # Close the loop
            fill='toself',
            name=model_name,
            line=dict(width=2),
        ))
    
    fig.update_layout(
        title=f"Radar Graph for {target_dataset[0]} - {radar_metric}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Assuming the metric is normalized between 0 and 1
            )
        ),
        showlegend=True
    )
    
    return fig

def get_radar_graph_for_metrics_per_class(model_results: dict[str, dict], target_class: tuple[str, int],
                                          target_dataset: tuple[str, list[str]]) -> Optional[go.Figure]:
    """
    Generates a Plotly radar graph for a specific class across multiple metrics for a model.
    """
    fig = go.Figure()
    
    categories = ["iou_score", "precision", "recall", "f1_score"]
    
    for model_name, model_result in model_results.items():
        dataset_metrics = model_result.get(target_dataset[0], {})
        target_class_metrics = next((cm for cm in dataset_metrics.get('class_metrics', []) if cm['class_name'] == target_class[0]), None)
        if not target_class_metrics:
            print_error_message(f"Class {target_class[0]} not found in dataset {target_dataset[0]}.")
            return None

        metric_values = [target_class_metrics.get(metric_name, 0.0) for metric_name in categories]
        fig.add_trace(go.Scatterpolar(
            r=metric_values,
            theta=[str.capitalize(c).replace('_', '-') for c in categories],
            fill='toself',
            name= f"{model_name}",
            ))

    fig.update_layout(
        title=f"Radar Graph for {target_class[0]} in {target_dataset[0]}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Assuming the metric is normalized between 0 and 1
            )
        ),
        showlegend=True
    )
    return fig

def get_auc_roc_pr_scores(results_test: ResultsTest, target_dataset: tuple[str, list[str]], target_classes: dict[str, int],
                          *,
                          subset_step: int = 1000) -> None:
    """
    Generates AUC-ROC and Precision-Recall curves for each target class for the model.
    """
    # target_classes = results_test.target_classes
    # if target_classes is None:
    #     print_error_message('Target classes are not provided in the config. Please check the config file.')
    #     return
    
    datasets = results_test.target_dataset_result_paths
    if target_dataset[0] not in datasets:
        print_error_message(f"Dataset {target_dataset[0]} not found in results.")
        return
    dataset_paths = datasets[target_dataset[0]]
    
    class_stats = []
    
    for target_class_name, target_class_id in target_classes.items():
        print_info_message(f"Processing AUC-ROC and Precision-Recall for class: {target_class_name} (ID: {target_class_id})")
        
        # Lists to collect predictions and ground truth
        all_probs = []
        all_targets = []
        
        for dataset_path in dataset_paths:
            print_info_message(f"Processing dataset: {dataset_path}")
            target_dir = os.path.join(dataset_path, 'target')
            softmax_dir = os.path.join(dataset_path, 'pred_logits')
            
            if not os.path.exists(softmax_dir):
                print_error_message('Softmax directory does not exist: {}'.format(softmax_dir))
                continue
            
            for softmax_file_name in tqdm(os.listdir(softmax_dir), total=len(os.listdir(softmax_dir)), desc=f"Processing softmax files for class {target_class_name}"):
                if not softmax_file_name.endswith('.npy'):
                    continue
                target_file_name = softmax_file_name.replace('pred_logits', 'target').replace('.npy', '.png')
                
                # Load predicted softmax probabilities
                probs = np.load(os.path.join(softmax_dir, softmax_file_name))  # shape: [C, H, W]
                prob_class = probs[target_class_id]  # shape: [H, W]
                
                # Load corresponding ground truth mask
                gt = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]

                # Create binary label mask for class
                binary_gt = (gt == target_class_id).astype(np.uint8)

                # Skip if ground truth mask is empty
                if np.sum(binary_gt) == 0: continue
                
                # Flatten both arrays
                all_probs.append(prob_class.flatten())
                all_targets.append(binary_gt.flatten())
            
        # Concatenate all data
        if len(all_probs) == 0 or len(all_targets) == 0:
            print_warning_message(f'No data found for class {target_class_id} ({target_class_name}). Skipping AUC-ROC and PR curves.')
            return
        y_scores = np.concatenate(all_probs)    # predicted probs for class
        y_true = np.concatenate(all_targets)    # binary ground truth for class
        
        # Compute AUC-ROC
        auc = roc_auc_score(y_true, y_scores)
        print(f"AUC-ROC for class {target_class_id} ({target_class_name}): {auc:.4f}")
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        subsampled_fpr, subsampled_tpr = fpr[::subset_step], tpr[::subset_step]  # Subsample for better visualization
        print("Lengths of fpr, tpr:", len(fpr), len(tpr))
        
        # Compute Average Precision (AP), i.e., area under PR curve
        ap = average_precision_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        print(f"Average Precision for class {target_class_id} ({target_class_name}): {ap:.4f}")

        subsampled_precision, subsampled_recall = precision[::subset_step], recall[::subset_step]  # Subsample for better visualization
        print("Lengths of precision, recall:", len(precision), len(recall))
        
        class_stats.append({
            'class_id': target_class_id,
            'class_name': target_class_name,
            'auc_roc': auc,
            'fpr': subsampled_fpr.tolist(),
            'tpr': subsampled_tpr.tolist(),
            'average_precision': ap,
            'precision': subsampled_precision.tolist(),
            'recall': subsampled_recall.tolist()
        })
        
    results_for_plot = {
        'model': results_test.model_name,
        'dataset': target_dataset[0],
        'class_metrics': class_stats
    }

    return results_for_plot

def get_auc_roc_curves(model_results_for_plot: dict[str, dict], target_dataset: tuple[str, list[str]], target_class: tuple[str, int]) -> go.Figure:
    """
    Plots AUC-ROC and Precision-Recall curves for each target class for the model.
    """
    class_name, class_id = target_class
    # Setup the figure
    fig = go.Figure()
    
    for model_name, metrics_for_plot in model_results_for_plot.items():
        # if target_dataset[0] not in metrics_for_plot:
        #     print_error_message(f"Dataset {target_dataset[0]} not found in model results for {model_name}.")
        #     continue

        # dataset_metrics = metrics_for_plot[target_dataset[0]]
        class_metrics = metrics_for_plot['class_metrics']
    
        class_metric = next((cm for cm in class_metrics if cm['class_name'] == class_name), None)
        if not class_metric:
            print_warning_message(f"Class {class_name} not found in dataset {target_dataset[0]} for model {model_name}.")
            continue
        
        auc_roc = class_metric.get('auc_roc', None)
        fpr = class_metric.get('fpr', None)
        tpr = class_metric.get('tpr', None)
        
        if auc_roc is None:
            print_warning_message(f"Metrics for class {class_name} not found in dataset {target_dataset[0]} for model {model_name}.")
            continue

        # Plot AUC-ROC curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC: {auc_roc:.4f})'))

    fig.update_layout(
        title=f"AUC-ROC Curve for {class_name} in {target_dataset[0]}",
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def get_pr_curves(model_results_for_plot: dict[str, dict], target_dataset: tuple[str, list[str]], target_class: tuple[str, int]) -> go.Figure:
    """
    Plots AUC-ROC and Precision-Recall curves for each target class for the model.
    """
    class_name, class_id = target_class
    # Setup the figure
    fig = go.Figure()
    
    for model_name, metrics_for_plot in model_results_for_plot.items():
        # if target_dataset[0] not in metrics_for_plot:
        #     print_error_message(f"Dataset {target_dataset[0]} not found in model results for {model_name}.")
        #     continue

        # dataset_metrics = metrics_for_plot[target_dataset[0]]
        class_metrics = metrics_for_plot['class_metrics']
    
        class_metric = next((cm for cm in class_metrics if cm['class_name'] == class_name), None)
        if not class_metric:
            print_warning_message(f"Class {class_name} not found in dataset {target_dataset[0]} for model {model_name}.")
            continue
        
        average_precision = class_metric.get('average_precision', None)
        precision = class_metric.get('precision', None)
        recall = class_metric.get('recall', None)

        if average_precision is None:
            print_warning_message(f"Metrics for class {class_name} not found in dataset {target_dataset[0]} for model {model_name}.")
            continue

        # Plot AUC-ROC curve
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{model_name} (AP: {average_precision:.4f})'))

    fig.update_layout(
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top'
        )
    )

    fig.update_layout(
        title=f"PR Curve for {class_name} in {target_dataset[0]}",
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    
    return fig

def save_fuse_segmentation_images(results_test_list: list[ResultsTest], target_dataset: tuple[str, list[str]], output_dir: str, *, 
        cmap: np.ndarray = common_cmap, img_original_size: tuple[int, int] = (1440, 1920)) -> None:
    """
    Fuses segmentation images from multiple results_test objects into a single CSV file.
    
    Args:
        results_test_list (list[ResultsTest]): List of ResultsTest objects containing the segmentation results.
        output_dir (str): Directory to save the fused segmentation images.
        cmap (np.ndarray): Colormap to use for visualizing the segmentation results.
    """
    if len(results_test_list) == 0:
        print_error_message("No results_test objects provided for fusion.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_results_test = results_test_list[-1]
    datasets = last_results_test.target_dataset_result_paths
    print(f"Available datasets in results: {list(datasets.keys())}")
    dataset_paths = datasets.get(target_dataset[0], [])
    if not dataset_paths:
        print_error_message(f"Dataset {target_dataset[0]} not found in results.")
        return
    
    first_dataset_path = dataset_paths[0]
    # Get input and ground truth from the first results_test object
    input_dir = os.path.join(first_dataset_path, 'input')
    target_dir = os.path.join(first_dataset_path, 'target')
    
    # Get all the prediction directories from all results_test objects
    pred_dirs = []
    for results_test in results_test_list:
        datasets = results_test.target_dataset_result_paths
        if target_dataset[0] not in datasets:
            continue
        dataset_paths = datasets[target_dataset[0]]
        for dataset_path in dataset_paths:
            pred_dir = os.path.join(dataset_path, 'pred')
            if os.path.exists(pred_dir):
                pred_dirs.append(pred_dir)
    
    input_files = sorted(os.listdir(input_dir))
    for input_file in tqdm(input_files, desc="Processing input files", total=len(input_files)):        
        # Load the original image
        input_image = Image.open(os.path.join(input_dir, input_file))
        input_image = input_image.resize(img_original_size, Image.BILINEAR)
        
        target_file = os.path.join(target_dir, os.path.basename(input_file).replace('input', 'target'))
        target_image = Image.open(target_file)
        target_image = target_image.resize(img_original_size, Image.BILINEAR)
        target_image = np.array(target_image)
        
        # Cmap the target image
        ## First, map the target image to the temporary ios_point_mapper_to_cocoStuff_custom_11_dict
        # target_image_new = np.zeros_like(target_image, dtype=np.uint8)
        # for key, value in ios_point_mapper_to_cocoStuff_custom_11_dict.items():
        #     target_image_new[target_image == key] = value
        # target_image = target_image_new
        target_image_colored = cmap[target_image].astype(np.uint8)  # shape: [H, W, 3]

        # Initialize a blank canvas for the fused segmentation
        fused_segmentation = np.zeros((img_original_size[1], img_original_size[0] * 3, len(pred_dirs)), dtype=np.uint8)

        # Process each prediction directory
        for pred_dir in pred_dirs:
            pred_file_path = os.path.join(pred_dir, os.path.basename(input_file).replace('input', 'pred'))
            if not os.path.exists(pred_file_path):
                print_warning_message(f"Prediction file {pred_file_path} does not exist. Skipping.")
                continue
            
            # Load the predicted mask
            pred_mask = Image.open(pred_file_path)  # shape: [H, W]
            pred_mask = pred_mask.resize(img_original_size, Image.BILINEAR)
            pred_mask = np.array(pred_mask)
            
            # Cmap the predicted mask to the number of classes
            ## Temp code to handle cityscapes
            if 'city' in pred_file_path:
                pred_mask_colored = cityscapes_cmap[pred_mask].astype(np.uint8)  # shape: [H, W, 3]
            else:
                pred_mask_colored = cmap[pred_mask].astype(np.uint8)  # shape: [H, W, 3]

            # Place the predicted mask in the correct col of the fused segmentation
            col_index = pred_dirs.index(pred_dir)
            fused_segmentation[:, col_index * img_original_size[0]:(col_index + 1) * img_original_size[0], :] = pred_mask_colored
            

        # Stack the input image, target image, and fused segmentation vertically
        stacked_image = np.hstack((np.array(input_image), target_image_colored, fused_segmentation))
        stacked_image_pil = Image.fromarray(stacked_image)
        
        # Save the stacked image
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        stacked_image_pil.save(output_file)
        print_info_message(f"Saved fused segmentation image: {output_file}")

def save_entropy_maps(results_test: ResultsTest, target_dataset: tuple[str, list[str]], output_dir: str) -> None:
    """
    Generates entropy maps for each input image using the predicted softmax probabilities.
    """
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)  # stability trick
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    datasets = results_test.target_dataset_result_paths
    if target_dataset[0] not in datasets:
        print_error_message(f"Dataset {target_dataset[0]} not found in results.")
        return
    dataset_paths = datasets[target_dataset[0]]
    
    for dataset_path in dataset_paths:
        print_info_message(f"Processing dataset: {dataset_path}")
        input_dir = os.path.join(dataset_path, 'input')
        logits_dir = os.path.join(dataset_path, 'pred_logits')
        target_dir = os.path.join(dataset_path, 'target_rgb')
        pred_dir = os.path.join(dataset_path, 'pred_rgb')

        if not os.path.exists(logits_dir):
            print_error_message('Logits directory does not exist: {}'.format(logits_dir))
            continue

        for logits_file_name in tqdm(os.listdir(logits_dir), total=len(os.listdir(logits_dir)), desc="Processing logits files"):
            if not logits_file_name.endswith('.npy'):
                continue

            # To stack with the input image, target_rgb, and pred_rgb
            input_file_name = logits_file_name.replace('pred_logits', 'input').replace('.npy', '.png')
            input_image_path = os.path.join(input_dir, input_file_name)
            if not os.path.exists(input_image_path):
                print_warning_message(f"Input image {input_image_path} does not exist. Skipping.")
                continue
            input_image = Image.open(input_image_path)
            input_image_array = np.array(input_image)
            
            target_rgb_file_name = logits_file_name.replace('pred_logits', 'target_rgb').replace('.npy', '.png')
            target_rgb_path = os.path.join(target_dir, target_rgb_file_name)
            if not os.path.exists(target_rgb_path):
                print_warning_message(f"Target RGB image {target_rgb_path} does not exist. Skipping.")
                continue
            target_rgb_image = Image.open(target_rgb_path)
            target_rgb_array = np.array(target_rgb_image)
            
            pred_rgb_file_name = logits_file_name.replace('pred_logits', 'pred_rgb').replace('.npy', '.png')
            pred_rgb_path = os.path.join(pred_dir, pred_rgb_file_name)
            if not os.path.exists(pred_rgb_path):
                print_warning_message(f"Predicted RGB image {pred_rgb_path} does not exist. Skipping.")
                continue
            pred_rgb_image = Image.open(pred_rgb_path)
            pred_rgb_array = np.array(pred_rgb_image)

            # Load predicted logits
            logits = np.load(os.path.join(logits_dir, logits_file_name))  # shape: [C, H, W]
            if len(logits.shape) != 3:
                print_warning_message(f"Skipping {logits_file_name} as it does not have the expected shape [C, H, W].")
                continue

            print("Shape and ranges of logits:", logits.shape, np.min(logits), np.max(logits))
            
            # Apply softmax to get probabilities
            logits = np.transpose(logits, (1, 2, 0))  # shape: [H, W, C]
            probs = softmax(logits, axis=-1)
            
            print("Shape and ranges of probabilities:", probs.shape, np.min(probs), np.max(probs))

            # Calculate entropy
            entropy = - np.sum(probs * np.log(probs + 1e-10), axis=-1)  # shape: [H, W]
            print("Shape and ranges of entropy:", entropy.shape, np.min(entropy), np.max(entropy))
            
            # Normalize entropy to [0, 255]
            entropy_normalized = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy)) * 255.0
            print("Ranges of entropy:", np.min(entropy), np.max(entropy), "Normalized ranges:", np.min(entropy_normalized), np.max(entropy_normalized))
            entropy_normalized = entropy_normalized.astype(np.uint8)  # shape: [H, W]
            
            # Stack the input, target_rgb, pred_rgb and entropy map horizontally
            entropy_colored = np.stack([entropy_normalized] * 3, axis=-1)  # shape: [H, W, 3]
            print("Shape of entropy_colored:", entropy_colored.shape)
            stacked_image = np.hstack((input_image_array, target_rgb_array, pred_rgb_array, entropy_colored))
            stacked_image_pil = Image.fromarray(stacked_image)
            
            # Save the stacked image
            output_file = os.path.join(output_dir, logits_file_name.replace('.npy', '.png'))
            stacked_image_pil.save(output_file)
            print_info_message(f"Saved entropy map for {logits_file_name} at {output_file}")