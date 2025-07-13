"""
This utility file contains functions used by the main train and test scripts for various purposes.
It includes using the arguments provided by the user to get the dataset, the model, optimizer, etc.
"""
import os
import numpy as np
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data

from utilities.print_utils import *
from config.config_schema import TestConfig

def get_dataset_config(args: TestConfig) -> tuple[data.Dataset, int, list, list]:
    dataset = None
    seg_classes = 0
    mean, std = None, None
    # read all the images in the folder
    if args.dataset == 'city':
        from data_loader.semantic_segmentation.cityscapes import CityscapesSegmentation, get_cityscapes_num_classes, get_cityscapes_mean_std
        from data_loader.semantic_segmentation.backup import CityscapesSegmentationTest
        dataset = CityscapesSegmentation(root=args.data_path, size=args.im_size, scale=args.s,
                                             coarse=False, train=(args.split == 'train'),
                                             mean=[0, 0, 0], std=[1, 1, 1],
                                             is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_cityscapes_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_cityscapes_mean_std()
        # seg_classes = 172  # Temporarily hardcoded for edge mapping dataset with coco stuff based training
    elif args.dataset == 'coco_stuff':
        from data_loader.semantic_segmentation.coco_stuff import COCOStuffSegmentation, get_cocoStuff_num_classes, get_cocoStuff_mean_std
        dataset = COCOStuffSegmentation(root_dir=args.data_path, split=args.split, is_training=False,
                                         scale=(args.s, args.s), crop_size=args.im_size,
                                         is_custom= args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_cocoStuff_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_cocoStuff_mean_std()
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        from data_loader.semantic_segmentation.edge_mapping import EdgeMappingSegmentation, get_edge_mapping_num_classes, get_edge_mapping_mean_std
        dataset = EdgeMappingSegmentation(root=args.data_path, train=False, scale=args.s, 
                                          size=args.im_size, ignore_idx=255,
                                            mean=[0, 0, 0], std=[1, 1, 1],
                                            is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_edge_mapping_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_edge_mapping_mean_std()
    elif args.dataset == 'ios_point_mapper':
        from data_loader.semantic_segmentation.ios_point_mapper import iOSPointMapperDataset, get_ios_point_mapper_num_classes, get_ios_point_mapper_mean_std
        dataset = iOSPointMapperDataset(root=args.data_path, train=False, scale=args.s,
                                        size=args.im_size, ignore_idx=255,
                                        mean=[0, 0, 0], std=[1, 1, 1],
                                        is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        seg_classes = get_ios_point_mapper_num_classes(is_custom=args.is_custom, custom_mapping_dict_key=args.custom_mapping_dict_key)
        mean, std = get_ios_point_mapper_mean_std()
    elif args.dataset == 'pascal':
        # print_error_message('Pascal dataset not yet supported in this script')
        raise NotImplementedError('Pascal dataset not implemented in this script')
    else:
        # print_error_message('{} dataset not yet supported'.format(args.dataset))
        raise NotImplementedError('Dataset {} not implemented'.format(args.dataset))
    return dataset, seg_classes, mean, std

def get_default_weights_test(args: TestConfig) -> str:
    if args.weights_test: # If weights_test is already set, do not override it
        return args.weights_test
    
    weights_test = ''
    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.weight_locations import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(weights_test):
            print_error_message('weight file does not exist: {}'.format(weights_test))

    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.weight_locations import model_weight_map

        model_key = '{}'.format(args.model)
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        weights_test = model_weight_map[model_key]['weights']
        if not os.path.isfile(weights_test):
            print_error_message('weight file does not exist: {}'.format(weights_test))
    else:
        # print_error_message('{} network not yet supported'.format(args.model))
        raise NotImplementedError('Network {} not implemented'.format(args.model))
    if not weights_test:
        # print_error_message('No weights found for model {} and dataset {}'.format(args.model, args.dataset))
        raise ValueError('No weights found for model {} and dataset {}'.format(args.model, args.dataset))
    return weights_test

def get_save_dir(args: TestConfig) -> str:
    savedir = ''
    # set-up results path
    if args.dataset == 'city':
        savedir = 'results_test/{}_{}_{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'ios_point_mapper':
        savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        savedir = 'results_test/{}_{}/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    elif args.dataset == 'coco_stuff':
        savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    else:
        # print_error_message('{} dataset not yet supported'.format(args.dataset))
        raise NotImplementedError('Dataset {} not implemented'.format(args.dataset))
    return savedir

def get_model_config(args: TestConfig, seg_classes: int = 0, mean=None, std=None) -> nn.Module:
    args.classes = args.classes if args.classes is not None else seg_classes
    args.mean = args.mean if args.mean is not None else mean
    args.std = args.std if args.std is not None else std
    if args.mean is None or args.std is None:
        # print_error_message('Mean and std values are not set. Please check the dataset configuration.')
        raise NotImplementedError('Mean and std values are not set.')
    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
        model = espnetv2_seg(args)
    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(n_classes=args.classes, aux_mode='eval')
    else:
        # print_error_message('{} network not yet supported'.format(args.model))
        raise NotImplementedError('Network {} not implemented'.format(args.model))
    return model

def get_cmap(args: TestConfig):
    from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_TRAIN_CMAP
    if args.dataset == 'city' or args.dataset == 'edge_mapping' or args.dataset == 'ios_point_mapper':
        cmap = CITYSCAPE_TRAIN_CMAP
    else:
        cmap = None
    return cmap

def prepare_save_images(args: TestConfig):
    """
    Prepare the directories for saving images.
    :param args: Namespace containing the arguments
    """
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(os.path.join(args.savedir, 'target'))
        os.makedirs(os.path.join(args.savedir, 'pred'))
        os.makedirs(os.path.join(args.savedir, 'input'))
        os.makedirs(os.path.join(args.savedir, 'pred_logits'))

def grayscale_tensor_to_rgb_tensor(tensor, cmap):
    """
    Convert a grayscale tensor to an RGB tensor using a colormap.
    :param tensor: Grayscale tensor of shape (C, H, W)
    :param cmap: Colormap to use for conversion (dict mapping grayscale values to RGB tuples)
    :return: RGB tensor of shape (3, H, W)
    """
    # Create an empty RGB tensor
    rgb_tensor = torch.zeros((3, tensor.shape[1], tensor.shape[2]), dtype=torch.uint8)
    # Iterate over the grayscale values and assign the corresponding RGB values
    for i in range(256):
        if i in cmap:
            rgb_tensor[0][tensor[0] == i] = cmap[i][0]
            rgb_tensor[1][tensor[0] == i] = cmap[i][1]
            rgb_tensor[2][tensor[0] == i] = cmap[i][2]

    return rgb_tensor

def save_images(args: TestConfig, input, target, output_prob, output, index, cmap=None):
    """
    Save the input, target and output images to the specified directory.
    :param args: Namespace containing the arguments
    :param input: Input tensor of shape (C, H, W)
    :param target: Target tensor of shape (H, W)
    :param output_prob: Output tensor of shape (C, H, W) with probabilities
    :param output: Output tensor of shape (H, W)
    :param index: Index of the image
    :param cmap: Colormap to use for conversion (dict mapping grayscale values to RGB tuples)
    """
    # Save the input image
    input_image = F.to_pil_image(input.cpu())
    input_image.save(os.path.join(args.savedir, 'input', 'input_{}.png'.format(index)))

    # Save the target image
    target = target.type(torch.ByteTensor)
    target_image = F.to_pil_image(target.cpu())  # Scale the target for better visibility
    target_image.save(os.path.join(args.savedir, 'target', 'target_{}.png'.format(index)))
    if cmap is not None:
        target_rgb_image = grayscale_tensor_to_rgb_tensor(target.unsqueeze(0), cmap)
        target_rgb_image = F.to_pil_image(target_rgb_image.cpu())
        target_rgb_image.save(os.path.join(args.savedir, 'target', 'target_rgb_{}.png'.format(index)))

    # Save the output probabilities as numpy array
    ## First convert to numpy array
    if args.save_output_probabilities:
        output_prob = output_prob.cpu().detach().numpy()
        np.save(os.path.join(args.savedir, 'pred_logits', 'pred_logits_{}.npy'.format(index)), output_prob)
    
    # Save the output image
    output = output.type(torch.ByteTensor)
    output_image = F.to_pil_image(output.cpu())  # Scale the output for better visibility
    output_image.save(os.path.join(args.savedir, 'pred', 'pred_{}.png'.format(index)))
    if cmap is not None:
        output_rgb_image = grayscale_tensor_to_rgb_tensor(output.unsqueeze(0), cmap)
        output_rgb_image = F.to_pil_image(output_rgb_image.cpu())
        output_rgb_image.save(os.path.join(args.savedir, 'pred', 'pred_rgb_{}.png'.format(index)))