"""
This experimental script is used to evaluate the performance of a semantic segmentation model on a dataset.
Any coreml model can be used for evaluation. The script uses the following metrics:
"""

import torch
import glob
import os
import json
import math
import numpy as np
from argparse import ArgumentParser
import time
from PIL import Image, ImageFile
import torch.utils
import torch.utils.data
from torchvision.transforms import functional as F, ToPILImage, ToTensor
from tqdm import tqdm
from utilities.print_utils import *
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
import coremltools as ct

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

def save_images(input_img: ImageFile, output_img: ImageFile, save_path, index):
    os.makedirs(save_path, exist_ok=True)

    # Map the grayscale images to RGB using a custom cityscapes color map
    palette = np.array([
        [255, 0, 0],   # Dynamic
        [0, 0, 255],    # Ground
        [128, 64, 128], # Road
        [244, 35, 232], # Sidewalk
        [250, 170, 30], # Parking
        [255, 85, 7],   # Rail track
        [119, 11, 32],  # Building
        [192, 192, 192],# Wall
        [204, 102, 255],# Fence
        [255, 153, 204],# Guard rail
        [153, 153, 153],# Bridge
        [238, 228, 198],# Tunnel
        [220, 220, 220],# Pole
        [220, 220, 220], # Polegroup
        [255, 204, 0],  # Traffic light
        [220, 220, 0],   # Traffic sign
        [107, 142, 35],  # Vegetation
        [152, 251, 152], # Terrain
        [70, 130, 180],  # Sky
        [220, 20, 60],   # Person
        [255, 0, 0],     # Rider
        [0, 0, 142],     # Car
        [0, 0, 70],      # Truck
        [0, 0, 255],     # Bus
        [0, 121, 215],   # Caravan
        [0, 0, 110],     # Trailer
        [0, 80, 100],    # Train
        [0, 0, 230],     # Motorcycle
        [119, 10, 32]    # Bicycle
    ])

    output_array = np.array(output_img)
    output_rgb_array = palette[output_array]
    output_rgb_img = Image.fromarray(output_rgb_array.astype(np.uint8))

    # Save the images
    input_img.save(os.path.join(save_path, 'input_{}.png'.format(index)))
    output_rgb_img.save(os.path.join(save_path, 'output_{}.png'.format(index)))
    # target_rgb_img.save(os.path.join(save_path, 'target_{}.png'.format(index)))


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

    # model.eval()
    # for i, imgName in tqdm(enumerate(zip(image_list, test_image_list)), total=len(image_list)):
    img_save_path = os.path.join(args.savedir, 'images')
    os.makedirs(img_save_path, exist_ok=True)
    for index, (inputs, target) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        # TODO: The way we are handling the dimensions, due to having a batch size, needs to be fixed.
        # The coreml model does not work with batches of tensors, but with single
        inputs: torch.Tensor = inputs.to(device=device)[0]
        target: torch.Tensor = target.to(device=device)[0]

        # img_out: torch.Tensor = model(inputs)
        # inputs_img = Image.fromarray(inputs.numpy())
        inputs_img = ToPILImage()(inputs)

        prediction = model.predict({"input": inputs_img})
        output_img = prediction['output']
        img_out = torch.from_numpy(output_img)
        # print("img_out shape: ", output_img.shape)
        
        # Save the images
        # save_images(inputs_img, output_img, img_save_path, index)

        # img_out = ToTensor()(output_img)
        

        # Get the metrics
        # for i in range(inputs.shape[0]):
        input_i = inputs.unsqueeze(0)
        target_i = target.unsqueeze(0)
        img_out_i = img_out.unsqueeze(0)
        # print("input_i shape: ", input_i.shape)
        # print("target_i shape: ", target_i.shape)
        # print("img_out_i shape: ", img_out_i.shape)

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
        dataset = CityscapesSegmentation(root=args.data_path, train=(args.split == "train"), size=args.im_size, scale=args.s,
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
    # dataset = torch.utils.data.Subset(dataset, range(10))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    print_info_message('Number of images in the dataset: {}'.format(len(dataset_loader.dataset)))

    args.classes = seg_classes - 1 if args.dataset == 'city' else seg_classes

    # load the model
    model = ct.models.MLModel(args.weights_test)

    # mdoel information
    # num_params = model_parameters(model)
    # flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]))
    # print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    # print_info_message('# of parameters: {}'.format(num_params))

    num_gpus = torch.cuda.device_count()
    # device = 'cuda' if num_gpus > 0 else 'cpu'
    device= 'cpu'
    # model = model.to(device=device)

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
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='data split')
    # parser.add_argument('--batch-size', type=int, default=1, help='list of batch sizes') # Keep it 1 for now
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--savedir', type=str, default='./results_segmentation_test', help='Location to save the results')

    args = parser.parse_args()
    args.batch_size = 1
    
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
