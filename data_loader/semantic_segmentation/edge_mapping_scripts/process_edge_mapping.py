"""
This script is used to process the Edge Mapping dataset.

The edge mapping dataset out-of-the-box contains rgb images along with corresponding depth images. 
The rgb images are stored in .png format. The depth images are stored in .npy format.
It also contains geospatial information of the images in a .csv file. 

The script processes the dataset to generate 3 splits: train, val, test.
For each split, the script will save the rgb image paths, depth image paths and the geospatial information 
    All the information is saved in separate split folders, as well as csv files.

Note: The paths are saved relative to the dataset directory.
"""
import os, glob, sys
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utilities.print_utils import *
from datetime import datetime

def check_empty_img(img_path: str): 
    # Reading Image 
    # You can give path to the  
    # image as first argument 
    image = cv2.imread(img_path) 
  
    # Checking if the image is empty or not 
    return image is None

def match_location_with_images(location_df: pd.DataFrame, image_files: list[str], 
                               *, location_identifier_column = 'sys_time', 
                               location_match_column = 'match_column', time_diff_column = 'time_diff'):
    """
    Function to match the geospatial information with the image files.
    The idea is to match the time stamp of the image (given by the file name) with 
    the time stamp in the geospatial information.

    The image file name is assumed to be in the format `YYYY_MM_DD_HH_MI_SS_mmmmmm_rgb.png`
    where `YYYY_MM_DD_HH_MI_SS` is the time stamp and `mmmmmm` is the millisecond part.
    This convention also applies to the depth file name: `YYYY_MM_DD_HH_MI_SS_mmmmmm_depth.npy`
    This convention also applies to the location identifier column in the geospatial information.

    Parameters:
    ----------
    location_df: pd.DataFrame
        Dataframe containing the geospatial information.

    image_files: list[str]
        List of image files.

    location_identifier_column: str
        Column name in the location_df which identifies the image.

    Returns:
    -------
    pd.DataFrame
        Dataframe containing the geospatial information of the image files.
        Ordered according to the image files.
    """
    # Get the image names and convert to date time format
    image_names = [datetime.strptime(
        '_'.join(os.path.basename(f).split('_')[:-1]), '%Y_%m_%d_%H_%M_%S_%f'
    ) for f in image_files]

    # Convert the location identifier column to date time format
    location_columns = location_df.columns.tolist()
    location_columns.append(time_diff_column)
    location_df['date_time'] = pd.to_datetime(
        location_df[location_identifier_column], format='%Y_%m_%d_%H_%M_%S_%f'
    )

    # Get the matching geospatial information
    image_locations = []
    for image_name in tqdm(image_names, desc='Matching location with images'):
        # Get the closest location information
        closest_location_index = (location_df['date_time'] - image_name).abs().idxmin()
        image_location = location_df.iloc[closest_location_index].copy()
        time_diff = (image_location['date_time'] - image_name)
        image_location.drop(labels=['date_time'], inplace=True)
        image_location = image_location.to_list()
        image_location.append(time_diff.total_seconds())
        image_locations.append(image_location)

    location_df.drop(columns=['date_time'], inplace=True)

    image_locations_df = pd.DataFrame(image_locations, columns=location_columns)

    return image_locations_df

def save_split(split, rgb_files, depth_files, location_df, output_path):
    """
    Function to save the split information.
    Parameters:
    ----------
    split: str
        Name of the split.
    rgb_files: list
        List of rgb image files.
    depth_files: list
        List of depth image files.
    location_df: pd.DataFrame
        Dataframe containing the geospatial information.
    output_path: str
        Path to the output folder.
    Returns:
    -------
    None
    """
    split_path = os.path.join(output_path, split)
    os.makedirs(split_path, exist_ok = True)

    # Save the image files
    rgb_folder = os.path.join(split_path, 'rgb')
    os.makedirs(rgb_folder, exist_ok = True)
    for f in rgb_files:
        os.system('cp {} {}'.format(f, rgb_folder))

    # Save the depth files
    depth_folder = os.path.join(split_path, 'depth')
    os.makedirs(depth_folder, exist_ok = True)
    for f in depth_files:
        os.system('cp {} {}'.format(f, depth_folder))

    # Save the location information
    location_df.to_csv(os.path.join(split_path, 'location.csv'))

def get_split_info(split: str, rgb_files: list[str], depth_files: list[str], location_df: pd.DataFrame, 
                   output_path: str,
                   *, match_column = 'match_column', location_identifier_column = 'sys_time'):
    """
    Function to save the split information.

    Parameters:
    ----------
    split: str
        Name of the split.

    rgb_files: list
        List of rgb image files.

    depth_files: list
        List of depth image files.

    location_df: pd.DataFrame
        Dataframe containing the geospatial information.

    output_path: str
        Path to the output folder.

    match_column: str
        Column name in the rgb_files which identifies the image.

    location_identifier_column: str
        Column name in the location_df which identifies the image.

    Returns:
    -------
    pd.DataFrame
        Dataframe containing the split information.
        Includes the rgb image paths, depth image paths and the geospatial information.
    """
    split_df = pd.DataFrame({'rgb': rgb_files, 'depth': depth_files})

    # Get the matching geospatial information
    split_df = pd.concat([split_df, location_df], axis=1)
    
    return split_df


def main(dataset_path: str, folder_patterns: dict, location_files: list[str], output_path: str, 
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
    # The image file name is assumed to be in the format `YYYY_MM_DD_HH_MI_SS_mmmmmm_rgb.png`
    rgb_files_path = os.path.join(dataset_path, 'rgb', folder_patterns['rgb'])
    # The depth file name is assumed to be in the format `YYYY_MM_DD_HH_MI_SS_mmmmmm_depth.npy`
    depth_files_path = os.path.join(dataset_path, 'depth', folder_patterns['depth'])

    rgb_files_temp = glob.glob(rgb_files_path)
    # rgb_files_temp = rgb_files_temp[:100] # MARK: For testing
    # Filter out empty images
    rgb_files = []
    for rgb_file in tqdm(rgb_files_temp, desc='Checking empty images'):
        if not check_empty_img(rgb_file):
            rgb_files.append(rgb_file)
    print("Number of empty images: {}".format(len(rgb_files_temp) - len(rgb_files)))

    depth_files = glob.glob(depth_files_path)
    # depth_files = depth_files[:100] # MARK: For testing

    print_info_message('{} rgb files found'.format(len(rgb_files)))
    print_info_message('{} depth files found'.format(len(depth_files)))

    # Load the geospatial information
    # MARK: For now, we will only use the first file
    location_df = pd.read_csv(os.path.join(dataset_path, location_files[0]), index_col = 0)
    # print(location_df.columns)

    # Split the dataset
    train_rgb_files, test_rgb_files = train_test_split(rgb_files, test_size = test_size, random_state = random_state)
    train_rgb_files, val_rgb_files = train_test_split(train_rgb_files, test_size = val_size/(1-test_size), random_state = random_state)

    print('RGB:: Train size: {}, Val size: {}, Test size: {}'.format(len(train_rgb_files), len(val_rgb_files), len(test_rgb_files)))

    # Get the corresponding depth files
    # MARK: This will fail if the word rgb is present in any parent folder's name
    train_depth_files = [f.replace('rgb', 'depth').replace('png', 'npy') for f in train_rgb_files]
    val_depth_files = [f.replace('rgb', 'depth').replace('png', 'npy') for f in val_rgb_files]
    test_depth_files = [f.replace('rgb', 'depth').replace('png', 'npy') for f in test_rgb_files]

    print("Depth:: Train size: {}, Val size: {}, Test size: {}".format(len(train_depth_files), len(val_depth_files), len(test_depth_files)))

    # Get the corresponding geospatial information
    train_location_df = match_location_with_images(location_df, train_rgb_files, location_identifier_column = location_identifier_column)
    val_location_df = match_location_with_images(location_df, val_rgb_files, location_identifier_column = location_identifier_column)
    test_location_df = match_location_with_images(location_df, test_rgb_files, location_identifier_column = location_identifier_column)
    print('Location:: Train size: {}, Val size: {}, Test size: {}'.format(len(train_location_df), len(val_location_df), len(test_location_df)))

    # Save the image, depth and location information in separate splits
    for split, files, depth_files, location_df in zip(['train', 'val', 'test'], 
                                                      [train_rgb_files, val_rgb_files, test_rgb_files], 
                                                      [train_depth_files, val_depth_files, test_depth_files],
                                                      [train_location_df, val_location_df, test_location_df]):
        print_info_message('Processing {} split'.format(split))
        
        # Transfer the files to the output folder
        save_split(split, files, depth_files, location_df, output_path)
        # print_info_message('Saved {} split at {}'.format(split, os.path.join(output_path, split)))
        
        # Save the csv file
        output_csv_path = os.path.join(output_path, '{}.csv'.format(split))
        # Instead of using the paths relative to the current directory, we will use the paths relative to the dataset directory
        data_files = [os.path.relpath(f, dataset_path) for f in files]
        depth_data_files = [os.path.relpath(f, dataset_path) for f in depth_files]
        split_df = get_split_info(split, data_files, depth_data_files, location_df, output_path)
        split_df.to_csv(output_csv_path, index = False)
        print_info_message('Saved {} split information at {}'.format(split, output_csv_path))


if __name__=="__main__":
    edge_mapping_path = '../../../datasets/edge_mapping/'
    image_folder_path = 'NorthSeattle_1118'
    dataset_path = os.path.join(edge_mapping_path, image_folder_path)
    output_path = os.path.join(edge_mapping_path, image_folder_path)
    folders = {'rgb': '*_rgb.png', 'depth': '*_depth.npy'} # Dictionary format: {folder_name: file_matching_pattern}
    location_files = ['north_seattle_11_18_meta_ref.csv', 'north_seattle_11_18_meta.csv']
    main(dataset_path, folders, location_files, output_path)