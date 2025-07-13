"""
This script processes results of a model to calculate various metrics such as AUC-ROC score, precision, recall, specificity, F1-score and Precision-Recall AUC.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
import os

base_path = 'results_test/results_ios_point_mapper/val/model_bisenetv2_ios_point_mapper/split_val/s_2.0_sc_1024_512/20250706-072008_city'
pred_dir = os.path.join(base_path, 'pred')
softmax_dir = os.path.join(base_path, 'pred_logits')
target_dir = os.path.join(base_path, 'target')

output_file = os.path.join(base_path, 'adhoc_metrics.txt')

# sidewalk, building, traffic sign, traffic light, pole
# target_classes = [22, 16, 10, 8, 21] # For version 35
# target_classes = [35, 19, 11, 8, 0] # For version 53
target_classes = [1, 2, 7, 6, 5] # For version Cityscapes

print(f"Processing softmax files in directory: {softmax_dir}")

for target_class in target_classes:
    tp, fp, tn, fn = 0, 0, 0, 0
    precision, recall, specificity, f1_score, iou_score = 0.0, 0.0, 0.0, 0.0, 0.0
    
    for pred_file_name in os.listdir(pred_dir):
        if not pred_file_name.endswith('.png'):
            continue
        if 'rgb' in pred_file_name or 'target' in pred_file_name:
            continue
        target_file_name = pred_file_name.replace('pred', 'target')
        
        # Load predicted mask
        pred_mask = np.array(Image.open(os.path.join(pred_dir, pred_file_name)))  # shape: [H, W]
        # Load corresponding ground truth mask
        gt_mask = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]
        
        # Create a mask that only contains the target class
        pred_mask_target = (pred_mask == target_class).astype(np.uint8)
        gt_mask_target = (gt_mask == target_class).astype(np.uint8)
        
        # Calculate True Positives, False Positives, True Negatives, False Negatives
        tp += np.sum((pred_mask_target == 1) & (gt_mask_target == 1))
        fp += np.sum((pred_mask_target == 1) & (gt_mask_target == 0))
        tn += np.sum((pred_mask_target == 0) & (gt_mask_target == 0))
        fn += np.sum((pred_mask_target == 0) & (gt_mask_target == 1))
        
    # Calculate metrics
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    print("precision:", precision, "recall:", recall, "specificity:", specificity)
    if tp + fn > 0 and tp + fp > 0: # Avoid division by zero
        f1_score = 2 * (precision * recall) / (precision + recall)
    if tp + fp + fn > 0:  # Avoid division by zero
        iou_score = tp / (tp + fp + fn)
    
    with open(output_file, 'a') as f:
        f.write(f"Metrics for class {target_class}:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1-score: {f1_score:.4f}\n")
        f.write(f"IoU: {iou_score:.4f}\n")
        f.write("\n")
    
    # Lists to collect predictions and ground truth
    # all_probs = []
    # all_targets = []

    # for softmax_file_name in os.listdir(softmax_dir):
    #     if not softmax_file_name.endswith('.npy'):
    #         continue
    #     target_file_name = softmax_file_name.replace('pred_logits', 'target').replace('.npy', '.png')
        
    #     # Load predicted softmax probabilities
    #     probs = np.load(os.path.join(softmax_dir, softmax_file_name))  # shape: [C, H, W]
    #     prob_class = probs[target_class]  # shape: [H, W]
        
    #     # Load corresponding ground truth mask
    #     gt = np.array(Image.open(os.path.join(target_dir, target_file_name)))  # shape: [H, W]

    #     # Create binary label mask for class
    #     binary_gt = (gt == target_class).astype(np.uint8)
        
    #     # Flatten both arrays
    #     all_probs.append(prob_class.flatten())
    #     all_targets.append(binary_gt.flatten())

    # # Concatenate all data
    # y_scores = np.concatenate(all_probs)    # predicted probs for class
    # y_true = np.concatenate(all_targets)    # binary ground truth for class

    # Compute AUC-ROC
    # auc = roc_auc_score(y_true, y_scores)
    # print(f"AUC-ROC for class {target_class}: {auc:.4f}")

    # from sklearn.metrics import roc_curve, precision_recall_curve
    # import matplotlib.pyplot as plt
    
    # Compute Average Precision (AP), i.e., area under PR curve
    # ap = average_precision_score(y_true, y_scores)
    # precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # print(f"Average Precision for class {target_class}: {ap:.4f}")

    # fpr, tpr, _ = roc_curve(y_true, y_scores)
    # plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curve for Class {target_class}')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'roc_curve_class_{target_class}.png')
    
    # # Clear plot
    # plt.clf()
    
    # Plot Precision-Recall curve
    # plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve for Class {target_class}')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'pr_curve_class_{target_class}.png')
    # plt.clf()