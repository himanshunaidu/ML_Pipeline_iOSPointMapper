"""
This script is used to generate summary statistics for the Edge Mapping dataset.
"""
import glob
import os
import sys
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utilities.print_utils import *

def main(dataset_path):
    folders = {'rgb': 'png', 'depth': 'npy'} # Dictionary format: {folder_name: file_extension}
    for folder, extension in folders.items():
        imageFiles = os.path.join(dataset_path, folder, '*.{}'.format(extension))
        files = glob.glob(imageFiles)
        print_info_message('{} files found for {} split'.format(len(files), folder))

if __name__=="__main__":
    edge_mapping_path = '../../../datasets/edge_mapping/'
    image_folder_path = 'NorthSeattle_1118'
    dataset_path = os.path.join(edge_mapping_path, image_folder_path)
    main(dataset_path)