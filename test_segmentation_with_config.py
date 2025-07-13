"""
This script is used to evaluate the performance of a semantic segmentation model on a dataset.
It tests the model on a specified dataset, computes various metrics (per class and overall), and saves the output images and metrics.
Only BiSeNetv2 and ESPNetv2 in this script.
"""

import torch
import glob
import os
import json
import math
from argparse import ArgumentParser, Namespace
import time
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
import torch.bin
from utilities.print_utils import *
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_TRAIN_CMAP
import numpy as np

from config import load_config
from config.config_schema import TestConfig
from eval.utils import AverageMeter
from eval.semantic_segmentation.custom_evaluation import CustomEvaluation
from eval.semantic_segmentation.metrics.old.persello import cityscapesIdToClassMap

def create_namespace_from_config(cfg) -> Namespace:
    """
    Create a Namespace object from the configuration dictionary.
    :param cfg: Configuration dictionary
    :return: Namespace object with the configuration parameters
    """
    args = Namespace()
    for key, value in cfg.items():
        setattr(args, key, value)
    return args

def set_default_weights_test(args: TestConfig):
    if args.weights_test: # If weights_test is already set, do not override it
        return
    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.weight_locations import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.weight_locations import model_weight_map

        model_key = '{}'.format(args.model)
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        args.weights_test = model_weight_map[model_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)

def set_save_dir(args: TestConfig):
    # set-up results path
    if args.dataset == 'city':
        args.savedir = 'results_test/{}_{}_{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        args.savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'ios_point_mapper':
        args.savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        args.savedir = 'results_test/{}_{}/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    elif args.dataset == 'coco_stuff':
        args.savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
        exit(-1)

def set_dataset_config(args: TestConfig):
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
        print_error_message('Pascal dataset not yet supported in this script')
        exit(-1)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
        exit(-1)
    return dataset, seg_classes, mean, std

def set_model_config(args: TestConfig, seg_classes: int = 0, mean=None, std=None):
    args.classes = args.classes if args.classes is not None else seg_classes
    args.mean = args.mean if args.mean is not None else mean
    args.std = args.std if args.std is not None else std
    if args.mean is None or args.std is None:
        print_error_message('Mean and std values are not set. Please check the dataset configuration.')
        exit(-1)
    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
        model = espnetv2_seg(args)
    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(n_classes=args.classes, aux_mode='eval')
    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)
    return model

def get_cmap(args: TestConfig):
    if args.dataset == 'city' or args.dataset == 'edge_mapping' or args.dataset == 'ios_point_mapper':
        cmap = CITYSCAPE_TRAIN_CMAP
    else:
        cmap = None
    return cmap

def preprocess_inputs(output, target, is_output_probabilities=True):
        """
        Preprocess the output and target tensors to get the predictions and ground truth.
        """
        if isinstance(output, tuple):
            output = output[0]

        if is_output_probabilities:
            _, pred = torch.max(output, 1)
        else:
            pred = output

        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()
        
        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        return pred, target

def data_transform(input, mean, std):
    input = F.normalize(input, mean, std)  # normalize the tensor
    return input

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
    # output_prob = output_prob.cpu().detach().numpy()
    # np.save(os.path.join(args.savedir, 'pred_logits', 'pred_logits_{}.npy'.format(index)), output_prob)
    
    # Save the output image
    output = output.type(torch.ByteTensor)
    output_image = F.to_pil_image(output.cpu())  # Scale the output for better visibility
    output_image.save(os.path.join(args.savedir, 'pred', 'pred_{}.png'.format(index)))
    if cmap is not None:
        output_rgb_image = grayscale_tensor_to_rgb_tensor(output.unsqueeze(0), cmap)
        output_rgb_image = F.to_pil_image(output_rgb_image.cpu())
        output_rgb_image.save(os.path.join(args.savedir, 'pred', 'pred_rgb_{}.png'.format(index)))

def evaluate(args: TestConfig, model, dataset_loader: torch.utils.data.DataLoader, device):
    im_size = tuple(args.im_size)

    cmap = get_cmap(args)   

    custom_eval = CustomEvaluation(num_classes=args.classes, max_regions=1024, is_output_probabilities=True, 
                                   idToClassMap=cityscapesIdToClassMap, args=args)
    
    # eval_classes = [i for i in range(args.classes)]
    # eval_classes = [22, 16, 10, 8, 21] # sidewalk, building, traffic sign, traffic light, pole
    # eval_classes = [35, 19, 11, 8, 0] # sidewalk, building, traffic sign, traffic light, pole
    eval_classes = [1, 2, 7, 6, 5]
    # Also get custom evaluation metrics per class
    # This will take in non-probability outputs to make the evaluation easier
    custom_eval_per_class = {
        eval_class: CustomEvaluation(num_classes=1, max_regions=1024, is_output_probabilities=False,
                         idToClassMap={0: 'road'}, miou_min_range=1, miou_max_range=2,
                         args=args) for eval_class in eval_classes
    }

    model.eval()
    # for i, imgName in tqdm(enumerate(zip(image_list, test_image_list)), total=len(image_list)):
    for index, (inputs, target) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        inputs: torch.Tensor = inputs.to(device=device)
        inputs = data_transform(inputs, args.mean, args.std)
        target: torch.Tensor = target.to(device=device)#.type(torch.ByteTensor)

        img_out: torch.Tensor = model(inputs)#.type(torch.ByteTensor)

        # Get the metrics
        for i in range(img_out.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            target_i = target[i].unsqueeze(0)
            img_out_i = img_out[i].unsqueeze(0)

            custom_eval.update(output=img_out_i, target=target_i)
            img_out_processed, target_processed = preprocess_inputs(img_out_i, target_i)

            for j in eval_classes:
                # Set class j to 0 in the output and target tensors
                # Set every other class to 255 in the output and target tensors
                img_out_processed_j = img_out_processed.clone()
                img_out_processed_j[img_out_processed_j != j] = 255
                img_out_processed_j[img_out_processed_j == j] = 0
                target_processed_j = target_processed.clone()
                target_processed_j[target_processed_j != j] = 255
                target_processed_j[target_processed_j == j] = 0
                custom_eval_per_class[j].update(output=img_out_processed_j, target=target_processed_j)

            # Save the images
            input_i_save = input_i.squeeze(0)
            target_i_save = target_i.squeeze(0)
            img_out_i_save = img_out_i.squeeze(0)
            img_out_processed_save = img_out_processed.squeeze(0)
            # print("Shapes: input {}, target {}, output {}, output processed {}".format(
            #     input_i_save.shape, target_i_save.shape, img_out_i_save.shape, img_out_processed_save.shape))
            save_images(args, input_i_save, target_i_save, img_out_i_save, img_out_processed_save, index*args.batch_size + i, cmap=cmap)

    # Get the metrics
    save_object = custom_eval.get_results()
    save_object['type'] = 'all'
    # for i in eval_classes:
    #     save_object_per_class = custom_eval_per_class[i].get_results()
    #     save_object_per_class['type'] = 'class_{}'.format(i)
    save_path = os.path.join(args.savedir, 'metrics.jsonl')
    with open(save_path, 'w') as f:
        json.dump(save_object, f)
        f.write('\n')
        for i in eval_classes:
            save_object_per_class = custom_eval_per_class[i].get_results()
            save_object_per_class['type'] = 'class_{}'.format(i)
            json.dump(save_object_per_class, f)
            f.write('\n')
    print_info_message('Metrics saved to {}'.format(save_path))

def main(args: TestConfig):
    dataset, seg_classes, mean, std = set_dataset_config(args)

    # Get a subset of the dataset
    dataset = torch.utils.data.Subset(dataset, range(10))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    print_info_message('Number of images in the dataset: {}'.format(len(dataset_loader.dataset)))

    model = set_model_config(args, seg_classes=seg_classes, mean=mean, std=std)

    # model information
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    print_info_message('# of parameters: {}'.format(num_params))

    if args.weights_test:
        print_info_message('Loading model weights')
        weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))
        model.load_state_dict(weight_dict, strict=False if args.model == 'bisenetv2' else True)
        print_info_message('Weight loaded successfully')
    else:
        print_error_message('weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(os.path.join(args.savedir, 'target'))
        os.makedirs(os.path.join(args.savedir, 'pred'))
        os.makedirs(os.path.join(args.savedir, 'input'))
        os.makedirs(os.path.join(args.savedir, 'pred_logits'))

    evaluate(args, model, dataset_loader, device=device)


if __name__ == '__main__':
    from config.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # config file
    parser.add_argument('--config', type=str, default='config/test_segmentation_bisenetv2_city.json', help='Path to the config file')

    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    set_default_weights_test(cfg) # args.weights_test will be set if not provided

    set_save_dir(cfg)  # args.savedir will be set based on the dataset and split
    if not os.path.isdir(cfg.savedir):
        os.makedirs(cfg.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    cfg.weights = ''
    cfg.im_size = tuple(cfg.im_size)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    cfg.savedir = '{}/model_{}_{}/split_{}/s_{}_sc_{}_{}/{}'.format(cfg.savedir, cfg.model, cfg.dataset, cfg.split,
                                                                     cfg.s, cfg.im_size[0], cfg.im_size[1], timestr)

    main(cfg)
