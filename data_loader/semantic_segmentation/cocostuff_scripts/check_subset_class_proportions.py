"""
This script analyzes the images in the specified directory to find class proportions in the annotations.
"""
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from collections import namedtuple

IMAGE_DIR_PATH = "labels/val2017"
IMAGES = glob.glob(IMAGE_DIR_PATH + "/*.png")

class_proportions = [0] * 256  # Assuming 256 classes for grayscale images (0-255)
total_pixels = 0

class_prop = [0.0668883508, 0.0036651629, 0.0143891645, 0.0114538200, 0.0035923707, 0.0238586474, 0.0166830874, 0.0128154126, 0.0025114286, 0.0018849533, 0.0038324564, 0.0000000000, 0.0034559698, 0.0014940386, 0.0065782532, 0.0005782021, 0.0008226748, 0.0027406700, 0.0027899659, 0.0008240594, 0.0020190306, 0.0028895826, 0.0001183708, 0.0008024319, 0.0016290005, 0.0000000000, 0.0005686225, 0.0056160145, 0.0000000000, 0.0000000000, 0.0010813539, 0.0000158490, 0.0016698482, 0.0001912219, 0.0000673350, 0.0000402322, 0.0000432369, 0.0007165510, 0.0001496751, 0.0000742832, 0.0011733323, 0.0004651844, 0.0002560274, 0.0004217665, 0.0000000000, 0.0000837248, 0.0001625127, 0.0000223628, 0.0000200958, 0.0000056088, 0.0006917967, 0.0013367636, 0.0002618580, 0.0000506896, 0.0002096570, 0.0001593949, 0.0001788029, 0.0005395943, 0.0008226227, 0.0003506417, 0.0005493911, 0.0027145695, 0.0005699868, 0.0011719047, 0.0004036663, 0.0000000000, 0.0033271800, 0.0000000000, 0.0000000000, 0.0001932355, 0.0000000000, 0.0000809373, 0.0000604886, 0.0000000000, 0.0000075546, 0.0000003371, 0.0000430875, 0.0000248267, 0.0005560112, 0.0000000000, 0.0000298450, 0.0003359896, 0.0000000000, 0.0001222940, 0.0017055915, 0.0002422803, 0.0000633507, 0.0012486858, 0.0000000000, 0.0000000000, 0.0000000000, 0.0019961361, 0.0000556875, 0.0011131288, 0.0030143994, 0.1003210416, 0.0095255235, 0.0003789369, 0.0025887165, 0.0008588797, 0.0003068098, 0.0052841405, 0.0000003235, 0.0002925582, 0.0028736250, 0.0355071374, 0.0001323713, 0.0000000000, 0.0004565800, 0.0000126430, 0.0122785695, 0.0031218852, 0.0107502785, 0.0002121322, 0.0037944731, 0.0024396454, 0.0026544134, 0.0012630619, 0.0014048000, 0.0087993726, 0.0005862521, 0.0004218751, 0.0022284997, 0.0388114772, 0.0035376945, 0.0040309840, 0.0012950494, 0.0191836921, 0.0012503442, 0.0006384124, 0.0000003439, 0.0189936014, 0.0005689483, 0.0001661146, 0.0045204452, 0.0000127652, 0.0001662323, 0.0002234086, 0.0012546113, 0.0758061643, 0.0000063419, 0.0041438433, 0.0029742305, 0.0043211303, 0.0101387743, 0.0009048452, 0.0061649169, 0.0041989946, 0.0656577946, 0.0012855536, 0.0038009213, 0.0001781332, 0.0000000000, 0.0069428321, 0.0087084236, 0.0001238190, 0.0932290321, 0.0056299268, 0.0083277310, 0.0006599109, 0.0008589702, 0.0009875473, 0.0017239406, 0.0026769007, 0.0011809276, 0.0025773292, 0.0020656861, 0.0000536897, 0.0834554045, 0.0003422229, 0.0068709177, 0.0165980753, 0.0049653890, 0.0012772841, 0.0033079191, 0.0008882179, 0.0024932922, 0.0022996539, 0.0000299061, 0.0000425106, 0.0059354990, 0.0019879321]
bg_class_prop = 0.0256344633

check_classes = {'road': 148, 'sidewalk': 139, 'building': [95, 127], 'traffic light': 131, 'pole': 9,
                 'traffic sign': [11, 12], 'vegetation': [93, 96, 128, 141, 168], 'terrain': [110, 123, 124, 125, 133, 135, 153, 158],
                 'background': 255}  # Background is mapped to 255
class_prop_map = {}
for class_name, class_ids in check_classes.items():
    if isinstance(class_ids, list):
        for class_id in class_ids:
            if class_id not in class_prop_map:
                class_prop_map[class_name] = 0
            class_prop_map[class_name] += class_prop[class_id] if class_id < len(class_prop) else 0
    else:
        class_prop_map[class_name] = class_prop[class_ids] if class_ids < len(class_prop) else 0
print("Class proportions from the precomputed values:")
for class_name, proportion in class_prop_map.items():
    print(f"{class_name}: {proportion:.10f}")
    
total_class_proportions = sum(class_prop_map.values())
print("Total class proportions from the precomputed values:", total_class_proportions)

# Get top classes not in check_classes
check_classes_values = set()
for class_ids in check_classes.values():
    if isinstance(class_ids, list):
        check_classes_values.update(class_ids)
    else:
        check_classes_values.add(class_ids)
top_classes = sorted(range(len(class_prop)), key=lambda i: class_prop[i], reverse=True)[:30]
print("Top classes not in check_classes:")
for class_id in top_classes:
    if class_id not in check_classes_values:
        print(f"Class {class_id}: {class_prop[class_id]:.10f}")

# top_classes = sky, person, clouds, bus, train, wall-concrete, car, truck, motorcycle, fence, playingfield

# road: 0.0656577946
# sidewalk: 0.0758061643
# building: 0.0191836921
# traffic light: 0.0189936014
# pole: 0.0018849533
# traffic sign: 0.0034559698
# vegetation: 0.0834554045
# terrain: 0.0083277310
# background: 0.0256344633

exit(-1)
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

