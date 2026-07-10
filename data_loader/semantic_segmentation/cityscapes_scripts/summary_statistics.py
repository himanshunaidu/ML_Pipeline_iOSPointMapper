"""
This script is used to generate summary statistics for the Cityscapes dataset.
"""
import glob
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utilities.print_utils import *

def analyse_label(label_path):
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    unique_labels, counts = np.unique(label, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))
    return label_distribution

def merge_distributions(distributions):
    merged_distribution = {}
    for dist in distributions:
        for label, count in dist.items():
            if label in merged_distribution:
                merged_distribution[label] += count
            else:
                merged_distribution[label] = count
    return merged_distribution

def main(cityscapesPath):
    splits = ['train', 'val', 'test']
    for split in splits:
        searchFine = os.path.join(cityscapesPath, "gtFine", split, "*", '*_labelTrainIds.png')
        filesFine = glob.glob(searchFine)
        print_info_message('{} files found for {} split'.format(len(filesFine), split))
        distributions = []
        for f in tqdm(filesFine, desc=f"Processing {split} split"):
            dist = analyse_label(f)
            distributions.append(dist)
        merged_distribution = merge_distributions(distributions)
        print_info_message('Label distribution for {} split: {}'.format(split, merged_distribution))
        

if __name__=="__main__":
    cityscapes_path = '../../../datasets/cityscapes/'
    main(cityscapes_path)