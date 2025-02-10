"""
This script is used to generate summary statistics for the Cityscapes dataset.
"""
import glob
import os
import sys
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utilities.print_utils import *

def main(cityscapesPath):
    splits = ['train', 'val', 'test']
    for split in splits:
        searchFine = os.path.join(cityscapesPath, "gtFine", split, "*", '*_labelTrainIds.png')
        filesFine = glob.glob(searchFine)
        print_info_message('{} files found for {} split'.format(len(filesFine), split))

if __name__=="__main__":
    cityscapes_path = '../../../datasets/cityscapes/'
    main(cityscapes_path)