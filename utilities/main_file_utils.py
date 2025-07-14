"""
This utility file contains functions used by the main train and test scripts for various purposes.
It includes using the arguments provided by the user to get the dataset, the model, optimizer, etc.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import torch
from PIL import Image
from tqdm import tqdm

from torch import nn
from torchvision.transforms import functional as F
from torch.utils import data

from utilities.print_utils import *
from config.config_schema import TestConfig

def get_dataset_config(args: TestConfig) -> tuple[data.Dataset, int, list, list]:
    dataset = None
    seg_classes = 0
    mean, std = None, None
    # read all the images in the folder
    if args.dataset == 'city':
        from data_loader.semantic_segmentation.cityscapes import CityscapesSegmentation, get_cityscapes_num_classes, get_cityscapes_mean_std
        from data_loader.semantic_segmentation.backup import CityscapesSegmentationTest
        dataset = CityscapesSegmentation(root=args.data_path, size=args.im_size, scale=args.s,
                                             coarse=False, train=(args.split == 'train'),
                                             mean=[0, 0, 0], std=[1, 1, 1],
                                             is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_cityscapes_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_cityscapes_mean_std()
        # seg_classes = 172  # Temporarily hardcoded for edge mapping dataset with coco stuff based training
    elif args.dataset == 'coco_stuff':
        from data_loader.semantic_segmentation.coco_stuff import COCOStuffSegmentation, get_cocoStuff_num_classes, get_cocoStuff_mean_std
        dataset = COCOStuffSegmentation(root_dir=args.data_path, split=args.split, is_training=False,
                                         scale=(args.s, args.s), crop_size=args.im_size,
                                         is_custom= args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_cocoStuff_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_cocoStuff_mean_std()
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        from data_loader.semantic_segmentation.edge_mapping import EdgeMappingSegmentation, get_edge_mapping_num_classes, get_edge_mapping_mean_std
        dataset = EdgeMappingSegmentation(root=args.data_path, train=False, scale=args.s, 
                                          size=args.im_size, ignore_idx=255,
                                            mean=[0, 0, 0], std=[1, 1, 1],
                                            is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_edge_mapping_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_edge_mapping_mean_std()
    elif args.dataset == 'ios_point_mapper':
        from data_loader.semantic_segmentation.ios_point_mapper import iOSPointMapperDataset, get_ios_point_mapper_num_classes, get_ios_point_mapper_mean_std
        dataset = iOSPointMapperDataset(root=args.data_path, train=False, scale=args.s,
                                        size=args.im_size, ignore_idx=255,
                                        mean=[0, 0, 0], std=[1, 1, 1],
                                        is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_ios_point_mapper_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_ios_point_mapper_mean_std()
    elif args.dataset == 'pascal':
        # print_error_message('Pascal dataset not yet supported in this script')
        raise NotImplementedError('Pascal dataset not implemented in this script')
    else:
        # print_error_message('{} dataset not yet supported'.format(args.dataset))
        raise NotImplementedError('Dataset {} not implemented'.format(args.dataset))
    return dataset, seg_classes, mean, std

def get_default_weights_test(args: TestConfig) -> str:
    if args.weights_test: # If weights_test is already set, do not override it
        return args.weights_test
    
    weights_test = ''
    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.weight_locations import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(weights_test):
            print_error_message('weight file does not exist: {}'.format(weights_test))

    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.weight_locations import model_weight_map

        model_key = '{}'.format(args.model)
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        weights_test = model_weight_map[model_key]['weights']
        if not os.path.isfile(weights_test):
            print_error_message('weight file does not exist: {}'.format(weights_test))
    else:
        # print_error_message('{} network not yet supported'.format(args.model))
        raise NotImplementedError('Network {} not implemented'.format(args.model))
    if not weights_test:
        # print_error_message('No weights found for model {} and dataset {}'.format(args.model, args.dataset))
        raise ValueError('No weights found for model {} and dataset {}'.format(args.model, args.dataset))
    return weights_test

def get_save_dir(args: TestConfig) -> str:
    savedir = ''
    # set-up results path
    if args.dataset == 'city':
        savedir = 'results_test/{}_{}_{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'ios_point_mapper':
        savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        savedir = 'results_test/{}_{}/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    elif args.dataset == 'coco_stuff':
        savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    else:
        # print_error_message('{} dataset not yet supported'.format(args.dataset))
        raise NotImplementedError('Dataset {} not implemented'.format(args.dataset))
    return savedir

def get_model_config(args: TestConfig, seg_classes: int = 0, mean=None, std=None) -> nn.Module:
    args.classes = args.classes if args.classes is not None else seg_classes
    args.mean = args.mean if args.mean is not None else mean
    args.std = args.std if args.std is not None else std
    if args.mean is None or args.std is None:
        # print_error_message('Mean and std values are not set. Please check the dataset configuration.')
        raise NotImplementedError('Mean and std values are not set.')
    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
        model = espnetv2_seg(args)
    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(n_classes=args.classes, aux_mode='eval')
    else:
        # print_error_message('{} network not yet supported'.format(args.model))
        raise NotImplementedError('Network {} not implemented'.format(args.model))
    return model

def get_cmap(args: TestConfig):
    if args.cmap is not None:
        return args.cmap
    from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_TRAIN_CMAP
    if args.dataset == 'city' or args.dataset == 'edge_mapping' or args.dataset == 'ios_point_mapper':
        cmap = CITYSCAPE_TRAIN_CMAP
    else:
        cmap = None
    return cmap

def prepare_save_images(args: TestConfig):
    """
    Prepare the directories for saving images.
    :param args: Namespace containing the arguments
    """
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(os.path.join(args.savedir, 'target'))
        os.makedirs(os.path.join(args.savedir, 'pred'))
        os.makedirs(os.path.join(args.savedir, 'target_rgb'))
        os.makedirs(os.path.join(args.savedir, 'pred_rgb'))
        os.makedirs(os.path.join(args.savedir, 'input'))
        os.makedirs(os.path.join(args.savedir, 'pred_logits'))

def grayscale_tensor_to_rgb_tensor(tensor, cmap):
    """
    Convert a grayscale tensor to an RGB tensor using a colormap.
    :param tensor: Grayscale tensor of shape (C, H, W)
    :param cmap: Colormap to use for conversion (dict mapping grayscale values to RGB tuples)
    :return: RGB tensor of shape (3, H, W)
    """
    # Create an empty RGB tensor
    rgb_tensor = torch.zeros((3, tensor.shape[1], tensor.shape[2]), dtype=torch.uint8)
    # Iterate over the grayscale values and assign the corresponding RGB values
    for i in range(256):
        if i in cmap:
            rgb_tensor[0][tensor[0] == i] = cmap[i][0]
            rgb_tensor[1][tensor[0] == i] = cmap[i][1]
            rgb_tensor[2][tensor[0] == i] = cmap[i][2]

    return rgb_tensor

def save_images(args: TestConfig, input, target, output_prob, output, index, cmap=None):
    """
    Save the input, target and output images to the specified directory.
    :param args: Namespace containing the arguments
    :param input: Input tensor of shape (C, H, W)
    :param target: Target tensor of shape (H, W)
    :param output_prob: Output tensor of shape (C, H, W) with probabilities
    :param output: Output tensor of shape (H, W)
    :param index: Index of the image
    :param cmap: Colormap to use for conversion (dict mapping grayscale values to RGB tuples)
    """
    # Save the input image
    input_image = F.to_pil_image(input.cpu())
    input_image.save(os.path.join(args.savedir, 'input', 'input_{}.png'.format(index)))

    # Save the target image
    target = target.type(torch.ByteTensor)
    target_image = F.to_pil_image(target.cpu())  # Scale the target for better visibility
    target_image.save(os.path.join(args.savedir, 'target', 'target_{}.png'.format(index)))
    if cmap is not None:
        target_rgb_image = grayscale_tensor_to_rgb_tensor(target.unsqueeze(0), cmap)
        target_rgb_image = F.to_pil_image(target_rgb_image.cpu())
        target_rgb_image.save(os.path.join(args.savedir, 'target_rgb', 'target_rgb_{}.png'.format(index)))

    # Save the output probabilities as numpy array
    ## First convert to numpy array
    if args.save_output_probabilities:
        output_prob = output_prob.cpu().detach().numpy()
        np.save(os.path.join(args.savedir, 'pred_logits', 'pred_logits_{}.npy'.format(index)), output_prob)
    
    # Save the output image
    output = output.type(torch.ByteTensor)
    output_image = F.to_pil_image(output.cpu())  # Scale the output for better visibility
    output_image.save(os.path.join(args.savedir, 'pred', 'pred_{}.png'.format(index)))
    if cmap is not None:
        output_rgb_image = grayscale_tensor_to_rgb_tensor(output.unsqueeze(0), cmap)
        output_rgb_image = F.to_pil_image(output_rgb_image.cpu())
        output_rgb_image.save(os.path.join(args.savedir, 'pred_rgb', 'pred_rgb_{}.png'.format(index)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
        "iou_score": "IoU",
        "pixel_count": "Pixel Count"
    })

    return df

def get_post_metrics(args: TestConfig) -> Tuple[str, str]:
    """
    Get the paths for saving post metrics.
    :param args: Namespace containing the arguments
    :return: Tuple containing the paths for post metrics and softmax directory
    
    NOTE: Should later move the core logic to eval folder.
    """
    output_file = os.path.join(args.savedir, 'post_metrics.json')
    target_classes = args.eval_classes
    if target_classes is None:
        print_error_message('Target classes are not provided in the config. Please check the config file.')
        return
    
    input_dir = os.path.join(args.savedir, 'input')
    pred_dir = os.path.join(args.savedir, 'pred')
    target_dir = os.path.join(args.savedir, 'target')
    
    class_stats = []
    total_pixels = 0
    weighted_iou_sum = 0.0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    all_iou_values = []
    all_precision_values = []
    all_recall_values = []

    for index, target_class in enumerate(target_classes):
        target_class_name = args.eval_class_names[index] \
            if args.eval_class_names is not None and index < len(args.eval_class_names) else ""
        
        tp, fp, tn, fn = 0, 0, 0, 0
        precision, recall, specificity, f1_score, iou_score = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for pred_file_name in tqdm(os.listdir(pred_dir), desc=f"Processing class {target_class}"):
            if not pred_file_name.endswith('.png'):
                continue
            target_file_name = pred_file_name.replace('pred', 'target')
            
            # Load predicted mask
            pred_mask = np.array(Image.open(os.path.join(pred_dir, pred_file_name)))  # shape: [H, W]
            # Load corresponding ground truth mask
            gt_mask = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]
            
            # Create a mask that only contains the target class
            pred_mask_target = (pred_mask == target_class).astype(np.uint8)
            gt_mask_target = (gt_mask == target_class).astype(np.uint8)
            
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
        'dataset': args.dataset,
        'model': args.model,
        'num_classes': args.classes,
        'im_size': args.im_size,
        'is_custom': args.is_custom,
        'custom_mapping_dict_key': args.custom_mapping_dict_key,
        'eval_classes': args.eval_classes,
        'eval_class_names': args.eval_class_names,
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
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print_info_message(f'Metrics saved to {output_file}')
    
    # Save to csv as well
    df = get_metrics_table(results)
    csv_file = output_file.replace('.json', '.csv')
    df.to_csv(csv_file, index=False)
    print_info_message(f'Metrics saved to {csv_file}: \n{df.to_markdown(index=False)}')
        
            
def get_post_viz(args: TestConfig):
    """
    Generate post-processing visualizations such as AUC-ROC and Precision-Recall curves.
    
    :param args: Namespace containing the arguments
    :return: None
    
    NOTE: Should later move the core logic to eval folder.
    """
    target_classes = args.eval_classes
    if target_classes is None:
        print_error_message('Target classes are not provided in the config. Please check the config file.')
        return
    if not args.save_output_probabilities:
        print_info_message('Skipping AUC-ROC and Precision-Recall curves as output probabilities are not saved.')
        return
    target_dir = os.path.join(args.savedir, 'target')
    softmax_dir = os.path.join(args.savedir, 'pred_logits')
    if not os.path.exists(softmax_dir):
        print_error_message('Softmax directory does not exist: {}'.format(softmax_dir))
        return
    
    from matplotlib import pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    auc_roc_dir = os.path.join(args.savedir, 'auc_roc')
    if not os.path.exists(auc_roc_dir): os.makedirs(auc_roc_dir)
    pr_curve_dir = os.path.join(args.savedir, 'pr_curve')
    if not os.path.exists(pr_curve_dir): os.makedirs(pr_curve_dir)
    
    for index, target_class in enumerate(target_classes):
        target_class_name = args.eval_class_names[index] \
            if args.eval_class_names is not None and index < len(args.eval_class_names) else ""
        
        # Lists to collect predictions and ground truth
        all_probs = []
        all_targets = []

        for softmax_file_name in os.listdir(softmax_dir):
            if not softmax_file_name.endswith('.npy'):
                continue
            target_file_name = softmax_file_name.replace('pred_logits', 'target').replace('.npy', '.png')
            
            # Load predicted softmax probabilities
            probs = np.load(os.path.join(softmax_dir, softmax_file_name))  # shape: [C, H, W]
            prob_class = probs[target_class]  # shape: [H, W]
            
            # Load corresponding ground truth mask
            gt = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]

            # Create binary label mask for class
            binary_gt = (gt == target_class).astype(np.uint8)
            
            # Skip if ground truth mask is empty
            if np.sum(binary_gt) == 0: continue
            
            # Flatten both arrays
            all_probs.append(prob_class.flatten())
            all_targets.append(binary_gt.flatten())

        # Concatenate all data
        y_scores = np.concatenate(all_probs)    # predicted probs for class
        y_true = np.concatenate(all_targets)    # binary ground truth for class

        # Compute AUC-ROC
        auc = roc_auc_score(y_true, y_scores)
        print(f"AUC-ROC for class {target_class} ({target_class_name}): {auc:.4f}")

        # Compute Average Precision (AP), i.e., area under PR curve
        ap = average_precision_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        print(f"Average Precision for class {target_class} ({target_class_name}): {ap:.4f}")

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {target_class} ({target_class_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(auc_roc_dir, f'roc_curve_class_{target_class}.png'))

        # Clear plot
        plt.clf()
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, label=f'AP = {ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for Class {target_class} ({target_class_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(pr_curve_dir, f'pr_curve_class_{target_class}.png'))
        plt.clf()
        
def recolor_images(args: TestConfig):
    input_dir = os.path.join(args.savedir, 'input')
    pred_dir = os.path.join(args.savedir, 'pred')
    pred_rgb_dir = os.path.join(args.savedir, 'pred_rgb')
    target_dir = os.path.join(args.savedir, 'target')
    target_rgb_dir = os.path.join(args.savedir, 'target_rgb')
    
    combined_dir = os.path.join(args.savedir, 'combined')
    if not os.path.exists(combined_dir): os.makedirs(combined_dir)
    
    cmap = args.cmap
    if cmap is None:
        print_error_message('Color map is not provided in the config. Please check the config file.')
        return
    target_classes = args.eval_classes
    if target_classes is None:
        print_error_message('Target classes are not provided in the config. Please check the config file.')
        return
    
    for pred_file_name in tqdm(os.listdir(pred_dir), desc="Recoloring images and saving as combined image for visualization"):
        if not pred_file_name.endswith('.png'):
            continue
        if 'rgb' in pred_file_name or 'target' in pred_file_name:
            continue
        target_file_name = pred_file_name.replace('pred', 'target')
        
        # Load predicted mask
        pred_mask = np.array(Image.open(os.path.join(pred_dir, pred_file_name)))  # shape: [H, W]
        
        # Load corresponding ground truth mask
        gt_mask = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]
        
        # Create a mask that only contains the target classes
        pred_mask_target = np.isin(pred_mask, target_classes).astype(np.uint8)
        gt_mask_target = np.isin(gt_mask, target_classes).astype(np.uint8)
        
        # Create RGB images for visualization
        pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        
        for i, class_id in enumerate(target_classes):
            pred_rgb[pred_mask == class_id] = cmap[class_id]
            gt_rgb[gt_mask == class_id] = cmap[class_id]
        
        # Save RGB images
        Image.fromarray(pred_rgb, mode='RGB').save(os.path.join(pred_rgb_dir, pred_file_name))
        Image.fromarray(gt_rgb, mode='RGB').save(os.path.join(target_rgb_dir, target_file_name))
        
        input_file_name = pred_file_name.replace('pred', 'input')
        input_image = Image.open(os.path.join(input_dir, input_file_name))
        # Resize input image to match the prediction size
        input_image = input_image.resize(pred_mask.shape[::-1], Image.BILINEAR)

        # Combine input, target, and prediction for visualization
        combined_image = Image.new('RGB', (input_image.width * 3, input_image.height))
        combined_image.paste(input_image, (0, 0))
        combined_image.paste(Image.fromarray(gt_rgb, mode='RGB'), (input_image.width, 0))
        combined_image.paste(Image.fromarray(pred_rgb, mode='RGB'), (input_image.width * 2, 0))
        combined_image.save(os.path.join(combined_dir, pred_file_name.replace('pred', 'combined')))
    
    print("Recoloring and saving images completed. Combined images are saved in: {}".format(combined_dir))
    return