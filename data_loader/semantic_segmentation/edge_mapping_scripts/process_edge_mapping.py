"""
This script is used to process the Edge Mapping dataset.

The edge mapping dataset out-of-the-box contains rgb images along with corresponding depth images. 
The rgb images are stored in .png format. The depth images are stored in .npy format.
It also contains geospatial information of the images in a .csv file. 

The script processes the dataset to generate 3 splits: train, val, test.
For each split, each image will have the following information:
    - rgb image
    - depth image
    - geospatial information (all stored in a single .csv file)
"""
import os, glob, sys
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utilities.print_utils import *



def main(dataset_path, folder_patterns, location_files, output_path, 
         *, random_state = 42, train_size = 0.7, val_size = 0.15, test_size = 0.15,
         location_identifier_column = 'sys_time'):
    """
    Main function to process the Edge Mapping dataset.

    Parameters:
    ----------
    dataset_path: str
        Path to the dataset folder.

    folder_patterns: dict
        Dictionary format: {folder_name: file_matching_pattern}
        For getting the rgb and depth images.

    location_files: list
        List of csv files containing geospatial information of the images.
        Typically, there are 2 files
    
    output_path: str
        Path to the output folder.

    random_state: int
        Random seed for splitting the dataset.

    train_size: float
        Fraction of the dataset to be used for training.

    val_size: float
        Fraction of the dataset to be used for validation.

    test_size: float
        Fraction of the dataset to be used for testing.

    location_identifier_column: str
        Column name in the location_df which identifies the image.

    Returns:
    -------
    None
    """
    # Load the rgb and depth images
    rgb_files_path = os.path.join(dataset_path, 'rgb', folder_patterns['rgb'])
    depth_files_path = os.path.join(dataset_path, 'depth', folder_patterns['depth'])

    rgb_files = glob.glob(rgb_files_path)
    depth_files = glob.glob(depth_files_path)
    print_info_message('{} rgb files found'.format(len(rgb_files)))
    print_info_message('{} depth files found'.format(len(depth_files)))

    # Load the geospatial information
    # MARK: For now, we will only use the first file
    location_df = pd.read_csv(os.path.join(dataset_path, location_files[0]), index_col = 0)
    print(location_df.columns)

    # Split the dataset
    train_files, test_files = train_test_split(rgb_files, test_size = test_size, random_state = random_state)
    train_files, val_files = train_test_split(train_files, test_size = val_size/(1-test_size), random_state = random_state)

    print('RGB:: Train size: {}, Val size: {}, Test size: {}'.format(len(train_files), len(val_files), len(test_files)))

    # Get the corresponding depth files
    # MARK: This will fail if the word rgb is present in any parent folder's name
    train_depth_files = [f.replace('rgb', 'depth') for f in train_files]
    val_depth_files = [f.replace('rgb', 'depth') for f in val_files]
    test_depth_files = [f.replace('rgb', 'depth') for f in test_files]

    print("Depth:: Train size: {}, Val size: {}, Test size: {}".format(len(train_depth_files), len(val_depth_files), len(test_depth_files)))
    print("First train file: ", os.path.basename(train_files[0]))

    # Get the corresponding geospatial information
    train_location_df = pd.DataFrame()
    val_location_df = pd.DataFrame()
    test_location_df = pd.DataFrame()



    

if __name__=="__main__":
    edge_mapping_path = '../../../datasets/edge_mapping/'
    image_folder_path = 'NorthSeattle_1118'
    dataset_path = os.path.join(edge_mapping_path, image_folder_path)
    output_path = os.path.join(edge_mapping_path, image_folder_path)
    folders = {'rgb': '*_rgb.png', 'depth': '*_depth.npy'} # Dictionary format: {folder_name: file_matching_pattern}
    location_files = ['north_seattle_11_18_meta_ref.csv', 'north_seattle_11_18_meta.csv']
    main(dataset_path, folders, location_files, output_path)