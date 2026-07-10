# This script will later copy images that contain specific class labels from the COCO dataset annotations and past them into a new directory.
# For now, it will just print the paths of the images that contain the specified class label.
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from collections import namedtuple

IMAGE_DIR_PATH = "labels/val2017"
IMAGES = glob.glob(IMAGE_DIR_PATH + "/*.png")

cocoStuff_continuous_7_dict_no_bg = {
    148:0, 139:1,
    95:2, 127:2, # building
    131:3, 9:4,
    11:5, 12:5, # traffic sign
    # The following classes are not needed for filtering, since they are supporting classes for providing context.
    # 93:6, 96:6, 128:6, 141:6, 168:6, # vegetation
    # 110:7, 123:7, 124:7, 125:7, 133:7, 135:7, 153:7, 158:7, # terrain
    # 255:8  # Background is mapped to 255, which is not used in the continuous labels.
}
class_labels = [label for label, new_label in cocoStuff_continuous_7_dict_no_bg.items()]
print(f"Class labels to check: {class_labels}")

for image_path in tqdm(IMAGES, desc="Processing images"):
    image_full_path = os.path.join(os.getcwd(), image_path)
    # Read the image as a grayscale image
    image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_full_path} could not be loaded. Please check the path and try again.")

    # Get the unique values in the image
    unique_values = np.unique(image)   
    
    # Check if any of the class labels are present in the unique values
    subset_labels = [label for label in class_labels if label in unique_values]
    if len(subset_labels) > 0:
        print("Unique values in image:", unique_values)
        print(f"Class labels {subset_labels} found in image: {image_full_path.replace('labels', 'images').replace('.png', '.jpg')}")
        print(f"Label Image: {image_full_path}")
        print("\n")
# This will print the paths of the images that contain any of the specified class labels.
# You can modify the script to copy these images to a new directory if needed.
# Note: The script assumes that the images are in grayscale format and that the class labels are represented by pixel values in the range of 0-255.
# If the images are in a different format or the class labels are represented differently, you may need to adjust the script accordingly.