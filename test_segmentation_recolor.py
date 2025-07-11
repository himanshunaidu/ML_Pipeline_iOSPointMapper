"""
This script processes results of a model to calculate various metrics such as AUC-ROC score.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
import os

base_path = 'results_test/results_ios_point_mapper/val/model_bisenetv2_ios_point_mapper/split_val/s_2.0_sc_640_640/20250706-072657_coco_ios_point_mapper'
input_dir = os.path.join(base_path, 'input')
softmax_dir = os.path.join(base_path, 'pred_logits')
pred_dir = os.path.join(base_path, 'pred')
pred_rgb_dir = os.path.join(base_path, 'pred_rgb')
target_dir = os.path.join(base_path, 'target')
target_rgb_dir = os.path.join(base_path, 'target_rgb')
combined_dir = os.path.join(base_path, 'combined')

input_mean = (0.3257, 0.3690, 0.3223)
input_std = (0.2112, 0.2148, 0.2115)

# road, sidewalk, building, traffic sign, traffic light, pole
target_classes = [27, 22, 16, 10, 8, 21] # For version 35
# target_classes = [41, 35, 19, 11, 8, 0] # For version 53
# target_classes = [0, 1, 2, 7, 6, 5] # For version Cityscapes

cmap = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [220, 220, 0], [250, 170, 30], [153, 153, 153]]

if not os.path.exists(pred_rgb_dir):
    os.makedirs(pred_rgb_dir)
if not os.path.exists(target_rgb_dir):
    os.makedirs(target_rgb_dir)
if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)

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
    
    # Create a mask that only contains the target classes
    pred_mask_target = np.isin(pred_mask, target_classes).astype(np.uint8)
    gt_mask_target = np.isin(gt_mask, target_classes).astype(np.uint8)
    
    # Create RGB images for visualization
    pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    
    for i, class_id in enumerate(target_classes):
        pred_rgb[pred_mask == class_id] = cmap[i]
        gt_rgb[gt_mask == class_id] = cmap[i]
    
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
    
    print(f"Processed {pred_file_name} and saved combined image.")