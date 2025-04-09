"""
This experimental script is used to evaluate the performance of a semantic segmentation model on a dataset.
Only BiSeNetv2 in this script.
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

from eval.utils import AverageMeter
from eval.semantic_segmentation.metrics.iou import IOU
from eval.semantic_segmentation.metrics.dice import Dice
from eval.semantic_segmentation.metrics.persello import Persello
from eval.semantic_segmentation.metrics.rom_rum import ROMRUM
from eval.semantic_segmentation.metrics.old.persello import segmentation_score_Persello as Persello_old, idToClassMap
from eval.semantic_segmentation.metrics.old.rom_rum import rom_rum as ROMRUM_old

def preprocess_inputs(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        _, pred = torch.max(output, 1)

        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()
        
        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        return pred, target


def evaluate(args, model, dataset_loader: torch.utils.data.DataLoader, device):
    im_size = tuple(args.im_size)

    losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    persello_over_list = []
    persello_under_list = []
    persello_over_meter = AverageMeter()
    persello_under_meter = AverageMeter()
    romrum_over_list = []
    romrum_under_list = []
    romrum_over_meter = AverageMeter()
    romrum_under_meter = AverageMeter()

    persello_old_over_list = []
    persello_old_under_list = []
    persello_old_over_meter = AverageMeter()
    persello_old_under_meter = AverageMeter()
    romrum_old_over_list = []
    romrum_old_under_list = []
    romrum_old_over_meter = AverageMeter()
    romrum_old_under_meter = AverageMeter()

    miou_class = IOU(num_classes=args.classes)
    persello_class = Persello(num_classes=args.classes, max_regions=1024)
    romrum_class = ROMRUM(num_classes=args.classes, max_regions=1024)

    # To record time taken for each metric
    start_time = 0
    miou_times = []
    persello_times = []
    romrum_times = []
    persello_old_times = []
    romrum_old_times = []

    # get color map for pascal dataset
    if args.dataset == 'pascal':
        from utilities.color_map import VOCColormap
        cmap = VOCColormap().get_color_map_voc()
    else:
        cmap = None

    model.eval()
    # for i, imgName in tqdm(enumerate(zip(image_list, test_image_list)), total=len(image_list)):
    for i, (inputs, target) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        inputs: torch.Tensor = inputs.to(device=device)
        target: torch.Tensor = target.to(device=device)

        img_out: torch.Tensor = model(inputs)

        # Get the metrics
        for i in range(img_out.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            target_i = target[i].unsqueeze(0)
            img_out_i = img_out[i].unsqueeze(0)

            start_time = time.time()
            inter, union = miou_class.get_iou(img_out_i, target_i)
            inter_meter.update(inter)
            union_meter.update(union)
            miou_times.append(time.time() - start_time)

            start_time = time.time()
            persello_over, persello_under = persello_class.get_persello(img_out_i, target_i)
            persello_over_list.append(persello_over)
            persello_under_list.append(persello_under)
            persello_over_meter.update(persello_over)
            persello_under_meter.update(persello_under)
            persello_times.append(time.time() - start_time)

            start_time = time.time()
            romrum_over, romrum_under = romrum_class.get_rom_rum(img_out_i, target_i)
            romrum_over_list.append(romrum_over)
            romrum_under_list.append(romrum_under)
            romrum_over_meter.update(romrum_over)
            romrum_under_meter.update(romrum_under)
            romrum_times.append(time.time() - start_time)

            # Get old persello metrics
            img_out_processed, target_processed = preprocess_inputs(model, img_out_i, target_i)
            img_out_processed, target_processed = img_out_processed.numpy()[0], target_processed.numpy()[0]

            start_time = time.time()
            persello_old_over, persello_old_under = 0, 0
            for class_id in idToClassMap.keys():
                persello_old_over_c, persello_old_under_c = Persello_old(target_processed, img_out_processed, class_id)
                persello_old_over += persello_old_over_c
                persello_old_under += persello_old_under_c
            persello_old_over_list.append(persello_old_over)
            persello_old_under_list.append(persello_old_under)
            persello_old_over_meter.update(persello_old_over/len(idToClassMap.keys()))
            persello_old_under_meter.update(persello_old_under/len(idToClassMap.keys()))
            persello_old_times.append(time.time() - start_time)

            start_time = time.time()
            romrum_old_over, romrum_old_under = 0, 0
            for class_id in idToClassMap.keys():
                romrum_old_over_c, romrum_old_under_c = ROMRUM_old(target_processed, img_out_processed, class_id)
                romrum_old_over += romrum_old_over_c
                romrum_old_under += romrum_old_under_c
            romrum_old_over_list.append(romrum_old_over)
            romrum_old_under_list.append(romrum_old_under)
            romrum_old_over_meter.update(math.tanh(romrum_old_over))#/len(idToClassMap.keys()))
            romrum_old_under_meter.update(math.tanh(romrum_old_under))#/len(idToClassMap.keys()))
            romrum_old_times.append(time.time() - start_time)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    dice = 2 * inter_meter.sum / (inter_meter.sum + union_meter.sum + 1e-10)
    miou = iou.mean()
    mdice = dice.mean()

    persello_over = persello_over_meter.sum / (persello_over_meter.count + 1e-10)
    persello_under = persello_under_meter.sum / (persello_under_meter.count + 1e-10)
    romrum_over = romrum_over_meter.sum / (romrum_over_meter.count + 1e-10)
    romrum_under = romrum_under_meter.sum / (romrum_under_meter.count + 1e-10)

    persello_old_over = persello_old_over_meter.sum / (persello_old_over_meter.count + 1e-10)
    persello_old_under = persello_old_under_meter.sum / (persello_old_under_meter.count + 1e-10)
    romrum_old_over = romrum_old_over_meter.sum / (romrum_old_over_meter.count + 1e-10)
    romrum_old_under = romrum_old_under_meter.sum / (romrum_old_under_meter.count + 1e-10)

    # Save the results
    save_object = {
        'mIoU': miou.item(),
        'mDice': mdice.item(),
        # 'Persello Over': persello_over_list,
        # 'Persello Under': persello_under_list,
        'ROM Over': romrum_over_list,
        'ROM Under': romrum_under_list,

        # 'Persello Old Over': persello_old_over_list,
        # 'Persello Old Under': persello_old_under_list,
        'ROM Old Over': romrum_old_over_list,
        'ROM Old Under': romrum_old_under_list,

        'mIoU time': miou_times,
        'Persello time': persello_times,
        'ROM time': romrum_times,
        'Persello Old time': persello_old_times,
        'ROM Old time': romrum_old_times,
    }
    save_path = os.path.join(args.savedir, 'metrics.json')
    with open(save_path, 'w') as f:
        json.dump(save_object, f, indent=4)
    print_info_message('Metrics saved to {}'.format(save_path))

def main(args):
    # read all the images in the folder
    if args.dataset == 'city':
        from data_loader.semantic_segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
        dataset = CityscapesSegmentation(root=args.data_path, train=False, size=args.im_size, scale=args.s,
                                             coarse=False)
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
    dataset = torch.utils.data.Subset(dataset, range(150))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    print_info_message('Number of images in the dataset: {}'.format(len(dataset_loader.dataset)))

    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        args.classes = seg_classes
        model = BiSeNetV2(n_classes=args.classes, aux_mode='pred')
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
        model.load_state_dict(weight_dict)
        print_info_message('Weight loaded successfully')
    else:
        print_error_message('weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

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
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
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
