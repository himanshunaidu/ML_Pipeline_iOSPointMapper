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

from utilities.main_file_utils import get_dataset_config, get_default_weights_test, get_save_dir, \
    get_model_config, get_cmap, prepare_save_images, save_images, recolor_images, \
    get_post_metrics, get_post_viz

from utilities.print_utils import *
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
import numpy as np

from config import load_config
from config.config_schema import TestConfig
from eval.utils import AverageMeter
from eval.semantic_segmentation.custom_evaluation import CustomEvaluation
from eval.semantic_segmentation.metrics.old.persello import cityscapesIdToClassMap

def preprocess_inputs(output, target, is_output_probabilities=True) -> tuple[torch.Tensor, torch.Tensor]:
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

def unnormalize(tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean).to(tensor.device).view(-1, 1, 1)
    std = torch.tensor(std).to(tensor.device).view(-1, 1, 1)
    return tensor * std + mean

def test(args: TestConfig, model, dataset_loader: torch.utils.data.DataLoader, device):
    args.cmap = get_cmap(args)   

    custom_eval = CustomEvaluation(num_classes=args.classes, max_regions=1024, is_output_probabilities=True, 
                                   idToClassMap=cityscapesIdToClassMap, args=args)
    
    # Also get custom evaluation metrics per class
    # This will take in non-probability outputs to make the evaluation easier
    custom_eval_per_class = {
        eval_class: CustomEvaluation(num_classes=1, max_regions=1024, is_output_probabilities=False,
                         idToClassMap={0: 'road'}, miou_min_range=1, miou_max_range=2,
                         args=args) for eval_class in args.eval_classes
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

            for j in args.eval_classes:
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
            input_i_save = unnormalize(input_i, args.mean, args.std).squeeze(0)
            target_i_save = target_i.squeeze(0)
            img_out_i_save = img_out_i.squeeze(0)
            img_out_processed_save = img_out_processed.squeeze(0)
            # print("Shapes: input {}, target {}, output {}, output processed {}".format(
            #     input_i_save.shape, target_i_save.shape, img_out_i_save.shape, img_out_processed_save.shape))
            save_images(args, input_i_save, target_i_save, img_out_i_save, img_out_processed_save, 
                        index*args.batch_size + i, cmap=args.cmap)

    # Get the metrics
    save_object = custom_eval.get_results()
    save_object['type'] = 'all'
    save_path = os.path.join(args.savedir, 'metrics.jsonl')
    with open(save_path, 'w') as f:
        json.dump(save_object, f)
        f.write('\n')
        for i in args.eval_classes:
            save_object_per_class = custom_eval_per_class[i].get_results()
            save_object_per_class['type'] = 'class_{}'.format(i)
            json.dump(save_object_per_class, f)
            f.write('\n')
    print_info_message('Metrics saved to {}'.format(save_path))

def evaluate(args: TestConfig):
    """
    Post-testing evaluation of the segmentation results.
    This function computes various metrics for each target class based on the predicted and ground truth masks.
    Metrics: mIoU, Precision, Recall, Specificity, F1-score
    Visualizations: AUC-ROC curves, Precision-Recall curves
    """
    get_post_metrics(args)
    get_post_viz(args)
    return

def main(args: TestConfig):
    dataset, seg_classes, mean, std = get_dataset_config(args)

    # Get a subset of the dataset
    # dataset = torch.utils.data.Subset(dataset, range(10))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    print_info_message('Number of images in the dataset: {}'.format(len(dataset_loader.dataset)))

    model = get_model_config(args, seg_classes=seg_classes, mean=mean, std=std)

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

    prepare_save_images(args)
    
    test(args, model, dataset_loader, device=device)
    recolor_images(args)
    evaluate(args)

if __name__ == '__main__':
    from config.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # config file
    parser.add_argument('--config', type=str, default='config/test_segmentation_bisenetv2_city.json', help='Path to the config file')

    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    cfg.weights_test = get_default_weights_test(cfg) # args.weights_test will be set if not provided

    cfg.savedir = get_save_dir(cfg)  # args.savedir will be set based on the dataset and split
    if not os.path.isdir(cfg.savedir):
        os.makedirs(cfg.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    cfg.weights = ''
    cfg.im_size = tuple(cfg.im_size)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    cfg.savedir = '{}/model_{}_{}/split_{}/s_{}_sc_{}_{}/{}'.format(cfg.savedir, cfg.model, cfg.dataset, cfg.split,
                                                                     cfg.s, cfg.im_size[0], cfg.im_size[1], timestr)

    main(cfg)
