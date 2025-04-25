"""
This experimental script is used to evaluate the performance of a semantic segmentation model on a dataset.
Only BiSeNetv2 and ESPNetv2 in this script.
"""

import torch
import glob
import os
import json
import math
from argparse import ArgumentParser
import time
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from utilities.print_utils import *
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_TRAIN_CMAP

from eval.utils import AverageMeter
from eval.semantic_segmentation.custom_evaluation import CustomEvaluation
from eval.semantic_segmentation.metrics.old.persello import cityscapesIdToClassMap

def preprocess_inputs(self, output, target, is_output_probabilities=True):
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

def evaluate(args, model, dataset_loader: torch.utils.data.DataLoader, device):
    im_size = tuple(args.im_size)

    # get color map for pascal dataset
    if args.dataset == 'pascal':
        from utilities.color_map import VOCColormap
        cmap = VOCColormap().get_color_map_voc()
    elif args.dataset == 'city':
        cmap = CITYSCAPE_TRAIN_CMAP
    else:
        cmap = None

    custom_eval = CustomEvaluation(num_classes=args.classes, max_regions=1024, is_output_probabilities=True, 
                                   idToClassMap=cityscapesIdToClassMap, args=args)

    model.eval()
    # for i, imgName in tqdm(enumerate(zip(image_list, test_image_list)), total=len(image_list)):
    for index, (inputs, target) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        inputs: torch.Tensor = inputs.to(device=device)
        target: torch.Tensor = target.to(device=device).type(torch.ByteTensor)

        img_out: torch.Tensor = model(inputs)#.type(torch.ByteTensor)

        # Get the metrics
        for i in range(img_out.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            target_i = target[i].unsqueeze(0)
            img_out_i = img_out[i].unsqueeze(0)

            custom_eval.update(output=img_out_i, target=target_i)

            # Save the images
            img_out_processed, target_processed = preprocess_inputs(model, img_out_i, target_i)
            target_i_image = F.to_pil_image(target_i.cpu()*10)
            target_i_image.save(os.path.join(args.savedir, 'target', 'target_{}.png'.format(index*args.batch_size + i)))
            target_i_rgb_image = grayscale_tensor_to_rgb_tensor(target_i, cmap)
            target_i_rgb_image = F.to_pil_image(target_i_rgb_image.cpu())
            target_i_rgb_image.save(os.path.join(args.savedir, 'target', 'target_rgb_{}.png'.format(index*args.batch_size + i)))
            img_out_image = F.to_pil_image(img_out_processed.cpu()*10)
            img_out_image.save(os.path.join(args.savedir, 'pred', 'pred_{}.png'.format(index*args.batch_size + i)))
            img_out_rgb_image = grayscale_tensor_to_rgb_tensor(img_out_processed, cmap)
            img_out_rgb_image = F.to_pil_image(img_out_rgb_image.cpu())
            img_out_rgb_image.save(os.path.join(args.savedir, 'pred', 'pred_rgb_{}.png'.format(index*args.batch_size + i)))

    # Get the metrics
    save_object = custom_eval.get_results()
    save_path = os.path.join(args.savedir, 'metrics.json')
    with open(save_path, 'w') as f:
        json.dump(save_object, f, indent=4)
    print_info_message('Metrics saved to {}'.format(save_path))

def main(args):
    # read all the images in the folder
    if args.dataset == 'city':
        from data_loader.semantic_segmentation.cityscapes import CityscapesSegmentationTest, CITYSCAPE_CLASS_LIST
        dataset = CityscapesSegmentationTest(root=args.data_path, size=args.im_size, scale=args.s,
                                             coarse=False, split=args.split)
        seg_classes = len(CITYSCAPE_CLASS_LIST)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        image_path = os.path.join(args.data_path, "rgb", "*.png")
        image_list = glob.glob(image_path)
        from data_loader.semantic_segmentation.edge_mapping import EDGE_MAPPING_CLASS_LIST
        seg_classes = len(EDGE_MAPPING_CLASS_LIST)
    elif args.dataset == 'pascal':
        from data_loader.semantic_segmentation.voc import VOC_CLASS_LIST
        seg_classes = len(VOC_CLASS_LIST)
        data_file = os.path.join(args.data_path, 'VOC2012', 'list', '{}.txt'.format(args.split))
        if not os.path.isfile(data_file):
            print_error_message('{} file does not exist'.format(data_file))
        image_list = []
        with open(data_file, 'r') as lines:
            for line in lines:
                rgb_img_loc = '{}/{}/{}'.format(args.data_path, 'VOC2012', line.split()[0])
                if not os.path.isfile(rgb_img_loc):
                    print_error_message('{} image file does not exist'.format(rgb_img_loc))
                image_list.append(rgb_img_loc)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
        exit(-1)


    # Get a subset of the dataset
    dataset = torch.utils.data.Subset(dataset, range(10))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    print_info_message('Number of images in the dataset: {}'.format(len(dataset_loader.dataset)))

    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
        args.mean = MEAN
        args.std = STD
    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        seg_classes = seg_classes - 1 if args.dataset == 'city' else seg_classes # Because the background class is not used in the model
        args.classes = seg_classes
        model = BiSeNetV2(n_classes=args.classes, aux_mode='eval')
        if args.dataset == 'city' or args.dataset == 'edge_mapping':
            args.mean = (0.3257, 0.3690, 0.3223)
            args.std = (0.2112, 0.2148, 0.2115)
        else:
            args.mean = MEAN
            args.std = STD
    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)

    # mdoel information
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

    evaluate(args, model, dataset_loader, device=device)


if __name__ == '__main__':
    from config.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # general details
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    # mdoel details
    parser.add_argument('--model', default="bisenetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="", help='Data directory')
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='data split')
    parser.add_argument('--batch-size', type=int, default=4, help='list of batch sizes')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--savedir', type=str, default='./results_segmentation_test', help='Location to save the results')

    args = parser.parse_args()

    if not args.weights_test:
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
    
    # set-up results path
    if args.dataset == 'city':
        args.savedir = 'results_test/{}_{}_{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        args.savedir = 'results_test/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        args.savedir = 'results_test/{}_{}/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''
    args.im_size = tuple(args.im_size)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.savedir = '{}/model_{}_{}/split_{}/s_{}_sc_{}_{}/{}'.format(args.savedir, args.model, args.dataset, args.split,
                                                                     args.s, args.im_size[0], args.im_size[1], timestr)

    main(args)
