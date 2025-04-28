"""
This script converts a ESPNetv2 PyTorch model to CoreML format for semantic segmentation.
Currently, it uses the old TorchScript method for conversion.
It is recommended to shift to the torch.export method for better performance.
"""
import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torchvision
import json
import cv2
from PIL import Image
from utilities.print_utils import *

from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
from transforms.semantic_segmentation.data_transforms import ToTensor
from transforms.semantic_segmentation.data_transforms import MEAN, STD

import coremltools as ct

class WrappedESPNetv2(nn.Module):
    def __init__(self, args):
        super(WrappedESPNetv2, self).__init__()
        self.model = espnetv2_seg(args=args)
        self.model.load_state_dict(torch.load(args.weight_path, map_location=torch.device('cpu')), strict=False)
        self.model.eval()

    def forward(self, x):
        res = self.model(x)
        # print('res shape:', res.shape)
        out = torch.argmax(res, dim=1, keepdim=True).float()
        # print('out shape:', out.shape)
        # out = out.float() / 255
        return out
    
if __name__ == '__main__':
    from config.general_details import segmentation_models, segmentation_datasets

    parser = argparse.ArgumentParser()
    # general details
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weight-path', default='', help='Pretrained weights directory.') # model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_512x256.pth
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="", help='Data directory') # datasets/cityscapes
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
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--outpath', dest='out_pth', type=str,
            default='./coreml/semantic_segmentation/model_zoo/')
    parser.add_argument('--img-path', dest='img_path', type=str, default='./datasets/custom_images/test.jpg',)
    args = parser.parse_args()

    args.weights = ''

    if args.dataset == 'city':
        from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_CLASS_LIST
        seg_classes = len(CITYSCAPE_CLASS_LIST)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        from data_loader.semantic_segmentation.edge_mapping import EDGE_MAPPING_CLASS_LIST
        seg_classes = len(EDGE_MAPPING_CLASS_LIST)
    elif args.dataset == 'pascal':
        from data_loader.semantic_segmentation.voc import VOC_CLASS_LIST
        seg_classes = len(VOC_CLASS_LIST)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
        exit(-1)
    args.classes = seg_classes

    # Prepare data
    to_tensor = ToTensor(
        # mean=(0.3257, 0.3690, 0.3223), # city, rgb
        # std=(0.2112, 0.2148, 0.2115),
        mean=(0.0, 0.0, 0.0), # placeholder
        std=(1.0, 1.0, 1.0),
    )
    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
    print('Loading image:', args.img_path)
    im = cv2.imread(args.img_path)#[:, :, ::-1]
    # Resize
    im = cv2.resize(im, args.im_size)
    empty_label = np.zeros(im.shape[:2], dtype=np.int64)
    im, label = to_tensor(rgb_img=im, label_img=empty_label)
    im = im.unsqueeze(0)
    
    # Prepare model
    torch_model = WrappedESPNetv2(args=args)
    torch_model = torch_model.to('cpu')
    torch_model.eval()
    # torch_model(im)
    # torch_model = torch_model.to('mps')
    # exit()
    traced_model = torch.jit.trace(torch_model, im)

    ml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=im.shape, scale=scale, bias=bias)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.GRAYSCALE)],
        # compute_precision=ct.precision.FLOAT16
        # minimum_deployment_target=ct.target.iOS13,
        # compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    ml_model_path = osp.join(args.out_pth, 'espnetv2_{}_{}_{}_{}.mlpackage'.format(args.dataset, args.num_classes, args.im_size[0], args.im_size[1]))
    ml_model.save(ml_model_path)
    print(f"Saved the model to {ml_model_path}")