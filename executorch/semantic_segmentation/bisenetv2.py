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

import coremltools as ct
from torch.export import export
from executorch.exir import to_edge

class WrappedBiSeNetv2(nn.Module):
    def __init__(self, n_cats, weight_path):
        super(WrappedBiSeNetv2, self).__init__()
        self.model = BiSeNetV2(n_classes=n_cats, aux_mode='eval')
        self.n_cats = n_cats
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('mps')), strict=False)
        self.model.eval()

    def forward(self, x):
        res = self.model(x)
        out = torch.argmax(res, dim=1, keepdim=True).float()
        return out
    
if __name__=="__main__":
    pass