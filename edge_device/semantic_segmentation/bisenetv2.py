"""
This script converts a BiSeNetv2 PyTorch model to CoreML format for semantic segmentation using ExecuTorch. 
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

# import coremltools as ct
from torch.export import export
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

class WrappedBiSeNetv2(nn.Module):
    def __init__(self, weight_path, n_cats, scale=1.0, bias=[0.0, 0.0, 0.0]):
        super(WrappedBiSeNetv2, self).__init__()
        self.model = BiSeNetV2(n_classes=n_cats, aux_mode='eval')
        self.n_cats = n_cats
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('mps')), strict=False)
        self.model = self.model.to(torch.device('mps'))
        self.model.eval()
        self.scale = torch.tensor(scale, dtype=torch.float32).to(torch.device('mps'))
        self.bias = torch.tensor(bias, dtype=torch.float32).reshape(1, 3, 1, 1).to(torch.device('mps'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the input
        x = x.to(torch.device('mps'))
        x = x * self.scale + self.bias
        res = self.model(x)
        out = torch.argmax(res, dim=1, keepdim=True).float()
        return out
    
if __name__=="__main__":
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
            default='./edge_device/semantic_segmentation/model_zoo/')
    parser.add_argument('--img-path', dest='img_path', type=str, default='./datasets/custom_images/test.jpg',)
    args = parser.parse_args()

    # Prepare data
    to_tensor = ToTensor(
        # mean=(0.3257, 0.3690, 0.3223), # city, rgb
        # std=(0.2112, 0.2148, 0.2115),
        mean=(0.0, 0.0, 0.0), # placeholder
        std=(1.0, 1.0, 1.0),
    )
    # NOTE: Weirdly enough, CoreML and Executorch have some difference in how they handle the scale and bias
    # Executorch, predictably enough, requires 1/255 scaling only if the input is scaled to 255
    # CoreML, if we use ImageType for input, seems to require the scaling even if the input is not scaled to 1
    # CoreML, if the input is scaled to 255, even then it requires the scaling
    # It is possible that internally, CoreML is scaling the input to 255 if the input is scaled to 1
    scale = 1/(0.2125)#*255.0)
    bias = [- 0.3257/(0.2112) , - 0.3690/(0.2148), - 0.3223/(0.2115)]
    print('Loading image:', args.img_path)
    im = cv2.imread(args.img_path)#[:, :, ::-1]
    # Resize
    im = cv2.resize(im, args.im_size)
    empty_label = np.zeros(im.shape[:2], dtype=np.int64)
    im, label = to_tensor(rgb_img=im, label_img=empty_label)
    im = im.unsqueeze(0)
    print('Image shape:', im.shape)

    # print('x sample pixel', im[0, :, 0, 0])
    # print('scale', scale)
    # print('bias sample pixel', torch.tensor(bias, dtype=torch.float32).reshape(1, 3, 1, 1)[0, :, 0, 0])
    # im_normalized = im * scale + torch.tensor(bias, dtype=torch.float32).reshape(1, 3, 1, 1)
    # print('Normalized x sample pixel', im_normalized[0, :, 0, 0])

    # Load the model
    torch_model = WrappedBiSeNetv2(weight_path=args.weight_path, n_cats=args.num_classes, scale=scale, bias=bias)
    torch_model.eval()

    torch_model = torch_model.to(torch.device('mps'))

    executorch_program = to_edge_transform_and_lower(
        export(torch_model, (im,)),
        partitioner=[CoreMLPartitioner()]
    ).to_executorch()

    ml_model_path = osp.join(args.out_pth, 'bisenetv2_{}_{}_{}.pte'.format(args.num_classes, args.im_size[0], args.im_size[1]))
    with(open(ml_model_path, 'wb')) as f:
        f.write(executorch_program.buffer)