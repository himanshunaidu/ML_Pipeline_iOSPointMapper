"""
This script converts a BiSeNetv2 PyTorch model to CoreML format for semantic segmentation.
Currently, it uses the old TorchScript method for conversion.
It is recommended to shift to the torch.export method for better performance.
"""
import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torchvision
import json
import cv2

from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2

import coremltools as ct

class WrappedBiSeNetv2(nn.Module):
    def __init__(self, n_cats, weight_path):
        super(WrappedBiSeNetv2, self).__init__()
        self.model = BiSeNetV2(n_classes=n_cats, aux_mode='eval')
        self.n_cats = n_cats
        self.model.load_state_dict(torch.load(weight_path), strict=False)
        self.model.eval()

    def forward(self, x):
        res = self.model(x)[0]
        out = torch.argmax(res, dim=1, keepdim=True).float()
        # out = out.float() / 255
        return out

if __name__ == '__main__':
    from config.general_details import segmentation_models, segmentation_schedulers, segmentation_loss_fns, \
        segmentation_datasets
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    # Only bisenetv2 is supported in this script
    # parser.add_argument('--model', default='bisenetv2', choices=segmentation_models,
    #                     help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--weight-path', type=str, default='./model/semantic_segmentation/model_zoo/bisenetv2/model_final_v2_city.pth',)
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=19, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--outpath', dest='out_pth', type=str,
            default='model.mlpackage')
    parser.add_argument('--img-path', dest='img_path', type=str, default='./datasets/custom_images/test.jpg',)
    args = parser.parse_args()