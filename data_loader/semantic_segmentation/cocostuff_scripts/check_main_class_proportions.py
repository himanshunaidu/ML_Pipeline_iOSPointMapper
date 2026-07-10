"""
This script analyzes the images in the specified directory to find class proportions in the annotations.
"""
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from collections import namedtuple

IMAGE_DIR_PATH = "labels/train2017"
IMAGES = glob.glob(IMAGE_DIR_PATH + "/*.png")

class_proportions = [0] * 182
total_pixels = 0

for image_path in tqdm(IMAGES, desc="Processing images"):
    image_full_path = os.path.join(os.getcwd(), image_path)
    # Read the image as a grayscale image
    image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_full_path} could not be loaded. Please check the path and try again.")

    # Get the number of pixels for each class
    unique, counts = np.unique(image, return_counts=True)
    class_counts = dict(zip(unique, counts))
    # Update the class proportions
    for class_id, count in class_counts.items():
        if class_id < len(class_proportions):
            class_proportions[class_id] += count
    # Update the total pixel count
    total_pixels += image.size
            
# Print the class proportions
for class_id, count in enumerate(class_proportions):
    if count > 0:
        print(f"Class {class_id}: {count} pixels")
print("Total pixels processed:", sum(class_proportions))
print("Unique classes found:", len([count for count in class_proportions if count > 0]))

# Normalize the class proportions
class_proportions = [(float(count) / total_pixels) for count in class_proportions]
class_proportions_str = ', '.join(f"{count:.10f}" for count in class_proportions)
print("Normalized class proportions:", class_proportions_str)