"""
This script generates mappings for the COCO-Stuff dataset, which is used for semantic segmentation tasks. 
The mappings file is a simple text file where each line is a pair of image and annotation file paths, separated by a comma.
"""
import os
import glob
import sys

DATASET_PATH = "../../../datasets/coco_stuff/coco/"
TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"

DESTINATION_PATH = "../../../datasets/coco_stuff/coco/"
TRAIN_MAPPINGS_FILE = "train.txt"
VAL_MAPPINGS_FILE = "val.txt"

def get_mappings(root_dir, split):
    """
    Generates a list of mappings for the given split (train or val).
    
    Args:
        root_dir (str): The root directory of the COCO-Stuff dataset.
        split (str): The dataset split, either 'train' or 'val'.
    
    Returns:
        list: A list of strings, each representing a mapping in the format 'image_path,annotation_path'.
    """
    image_dir = os.path.join(root_dir, "images", split)
    annotation_dir = os.path.join(root_dir, "annotations", split)

    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.png"))

    image_files.sort()
    annotation_files.sort()

    if len(image_files) != len(annotation_files):
        print(f"Warning: Number of images ({len(image_files)}) and annotations ({len(annotation_files)}) do not match for {split} split.")

    mappings = []
    for img_file, ann_file in zip(image_files, annotation_files):
        if os.path.basename(img_file).replace('.jpg', '.png') != os.path.basename(ann_file):
            print(f"Warning: Image file {img_file} does not match annotation file {ann_file}. Skipping this pair.")
            exit(-1)
        img_file_rel = os.path.relpath(img_file, root_dir)
        ann_file_rel = os.path.relpath(ann_file, root_dir)
        mappings.append(f"{img_file_rel},{ann_file_rel}")

    return mappings

def save_mappings(mappings, output_file):
    """
    Saves the mappings to a text file.
    
    Args:
        mappings (list): A list of mappings to save.
        output_file (str): The path to the output text file.
    """
    with open(output_file, 'w') as f:
        for mapping in mappings:
            f.write(mapping + '\n')
    print(f"Saved {len(mappings)} mappings to {output_file}")
    
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from utilities.print_utils import *

    # Generate and save mappings for the training split
    train_mappings = get_mappings(DATASET_PATH, TRAIN_SPLIT)
    print_info_message(f"Generated {len(train_mappings)} training mappings.")
    save_mappings(train_mappings, os.path.join(DESTINATION_PATH, TRAIN_MAPPINGS_FILE))

    # Generate and save mappings for the validation split
    val_mappings = get_mappings(DATASET_PATH, VAL_SPLIT)
    print_info_message(f"Generated {len(val_mappings)} validation mappings.")
    save_mappings(val_mappings, os.path.join(DESTINATION_PATH, VAL_MAPPINGS_FILE))