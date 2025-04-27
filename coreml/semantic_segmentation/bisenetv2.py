"""
This script converts a BiSeNetv2 PyTorch model to CoreML format for semantic segmentation.
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

from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
from transforms.semantic_segmentation.data_transforms import ToTensor

import coremltools as ct

class WrappedBiSeNetv2(nn.Module):
    def __init__(self, n_cats, weight_path):
        super(WrappedBiSeNetv2, self).__init__()
        self.model = BiSeNetV2(n_classes=n_cats, aux_mode='eval')
        self.n_cats = n_cats
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('mps')), strict=False)
        self.model.eval()

    def forward(self, x):
        res = self.model(x)
        # print('res shape:', res.shape)
        out = torch.argmax(res, dim=1, keepdim=True).float()
        # print('out shape:', out.shape)
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
                        help='Cityscapes classes.')
    parser.add_argument('--im-size', default=[1024, 512], type=int, nargs='+',
                        help='Input image size. The model will be resized to this size.')
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--outpath', dest='out_pth', type=str,
            default='./coreml/semantic_segmentation/model_zoo/')
    parser.add_argument('--img-path', dest='img_path', type=str, default='./datasets/custom_images/test.jpg',)
    args = parser.parse_args()

    # Prepare data
    to_tensor = ToTensor(
        # mean=(0.3257, 0.3690, 0.3223), # city, rgb
        # std=(0.2112, 0.2148, 0.2115),
        mean=(0.0, 0.0, 0.0), # placeholder
        std=(1.0, 1.0, 1.0),
    )
    scale = 1/(0.2125*255.0)
    bias = [- 0.3257/(0.2112) , - 0.3690/(0.2148), - 0.3223/(0.2115)]
    print('Loading image:', args.img_path)
    im = cv2.imread(args.img_path)#[:, :, ::-1]
    # Resize
    im = cv2.resize(im, args.im_size)
    empty_label = np.zeros(im.shape[:2], dtype=np.int64)
    im, label = to_tensor(rgb_img=im, label_img=empty_label)
    im = im.unsqueeze(0)

    # Prepare model
    torch_model = WrappedBiSeNetv2(n_cats=args.num_classes, weight_path=args.weight_path)
    torch_model.eval()
    # torch_model(im)
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

    ml_model_path = osp.join(args.out_pth, 'bisenetv2_{}_{}_{}.mlpackage'.format(args.num_classes, args.im_size[0], args.im_size[1]))
    ml_model.save(ml_model_path)
    print(f"Saved the model to {ml_model_path}")