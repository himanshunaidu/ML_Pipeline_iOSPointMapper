# This script will later copy images that contain specific class labels from the COCO dataset annotations and paste them into a new directory.
# For now, it will just print the paths of the images that contain the specified class label.
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from collections import namedtuple

DATASET_PATH = "../../../datasets/coco_stuff/coco"

IMAGE_DIR_PATH = os.path.join(DATASET_PATH, "images/val2017")
LABEL_DIR_PATH = os.path.join(DATASET_PATH, "annotations/val2017")
LABELS = glob.glob(LABEL_DIR_PATH + "/*.png")

TARGET_IMAGE_DIR_PATH = os.path.join(DATASET_PATH, "coco_subset/images/val2017")
TARGET_LABEL_DIR_PATH = os.path.join(DATASET_PATH, "coco_subset/annotations/val2017")
print(f"Target image directory: {TARGET_IMAGE_DIR_PATH}")
print(f"Target label directory: {TARGET_LABEL_DIR_PATH}")

if not os.path.exists(TARGET_IMAGE_DIR_PATH):
    os.makedirs(TARGET_IMAGE_DIR_PATH)
if not os.path.exists(TARGET_LABEL_DIR_PATH):
    os.makedirs(TARGET_LABEL_DIR_PATH)

cocoStuff_continuous_7_dict_no_bg = {
    148:0, 139:1,
    95:2, 127:2, # building
    # 131:3, # Even metal is pretty ambiguous for our use case (don't have a dedicated class for pole)
    9:4,
    11:5, 12:5, # traffic sign
    # The following classes are not needed for filtering, since they are supporting classes for providing context.
    # 93:6, 96:6, 128:6, 141:6, 168:6, # vegetation
    # 110:7, 123:7, 124:7, 125:7, 133:7, 135:7, 153:7, 158:7, # terrain
    # 255:8  # Background is mapped to 255, which is not used in the continuous labels.
}
class_labels = [label for label, new_label in cocoStuff_continuous_7_dict_no_bg.items()]
print(f"Class labels to check: {class_labels}")

COUNT = 10
for label_path in tqdm(LABELS, desc="Processing labels"):
    image_path = os.path.join(IMAGE_DIR_PATH, os.path.basename(label_path).replace('.png', '.jpg'))
    # print(f"Processing label: {os.path.basename(label_path)}\n and image: {os.path.basename(image_path)}")

    target_label_path = os.path.join(TARGET_LABEL_DIR_PATH, os.path.basename(label_path))
    target_image_path = os.path.join(TARGET_IMAGE_DIR_PATH, os.path.basename(image_path))
    # print(f"Target label path: {target_label_path}\n and target image path: {target_image_path}")
    
    # Read the label image as a grayscale image
    label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # Check if the label image was loaded successfully
    if label_image is None:
        raise ValueError(f"Label image at path {label_path} could not be loaded. Please check the path and try again.")

    # Get the unique values in the label image
    unique_values = np.unique(label_image)
    # Check if any of the class labels are present in the unique values
    subset_labels = [label for label in class_labels if label in unique_values]
    if len(subset_labels) == 0:
        continue  # Skip if no relevant class labels are found
    print("Unique values in label image:", unique_values)
    print(f"Class labels {subset_labels} found in label image: {label_path}")
    
    if os.path.exists(image_path):
        cv2.imwrite(target_image_path, cv2.imread(image_path))
        cv2.imwrite(target_label_path, label_image)
        print(f"Copied {os.path.basename(image_path)} to {TARGET_IMAGE_DIR_PATH} and {TARGET_LABEL_DIR_PATH}")
    else:
        print(f"Image {os.path.basename(image_path)} does not exist.")
    

# This will print the paths of the images that contain any of the specified class labels.
# You can modify the script to copy these images to a new directory if needed.
# Note: The script assumes that the images are in grayscale format and that the class labels are represented by pixel values in the range of 0-255.
# If the images are in a different format or the class labels are represented differently, you may need to adjust the script accordingly.