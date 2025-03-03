"""
This script is used to generate summary statistics for the Edge Mapping dataset.
"""
import glob
import os
import sys
from PIL import Image
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utilities.print_utils import *

def main(dataset_path):
    folders = {'rgb': '*_rgb.png', 'depth': '*_depth.npy'} # Dictionary format: {folder_name: file_matching_pattern}
    for folder, match_format in folders.items():
        imageFiles = os.path.join(dataset_path, folder, match_format)
        files = glob.glob(imageFiles)
        print_info_message('{} files found for {} split'.format(len(files), folder))

        if 'png' in match_format:
            image_sizes = []
            for file in files[:10]:
                image = Image.open(file)
                image_sizes.append(image.size)
            print_info_message('Image sizes for {} split: {}'.format(folder, set(image_sizes)))

if __name__=="__main__":
    edge_mapping_path = '../../../datasets/edge_mapping/'
    image_folder_path = 'NorthSeattle_1118'
    dataset_path = os.path.join(edge_mapping_path, image_folder_path)
    main(dataset_path)