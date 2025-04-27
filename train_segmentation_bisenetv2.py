"""
This script is used to train the BiSeNetV2 model for semantic segmentation.

Currently, this script is very similar to the original implementation.
Thus it utilizes a config file from the config directory to set the training parameters.
"""
import sys
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from config import set_cfg_from_file
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from train.semantic_segmentation.loss_functions.ohem_ce_loss import OhemCELoss
from train.lr_scheduler_2 import WarmupPolyLrScheduler
from eval.utils import AverageMeter, TimeMeter
# from utilities.print_utils import *
from utilities.log_utils import setup_logger, log_msg

## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2_coco.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)

def set_model(lb_ignore=255):
    logger = logging.getLogger()
    if cfg.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        seg_classes = seg_classes - 1 if cfg.dataset == 'city' else seg_classes # Because the background class is not used in the model
        cfg.classes = seg_classes
        model = BiSeNetV2(n_classes=cfg.classes, aux_mode='eval')
        if cfg.dataset == 'city' or cfg.dataset == 'edge_mapping':
            cfg.mean = (0.3257, 0.3690, 0.3223)
            cfg.std = (0.2112, 0.2148, 0.2115)
        else:
            cfg.mean = MEAN
            cfg.std = STD
    else:
        logger.error('{} network not yet supported'.format(cfg.model))
        exit(-1)
    model.cuda()
    model.train()
    return model

def set_criteria(lb_ignore=255):
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)]
    return criteria_pre, criteria_aux

def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim

def set_model_dist(net):
    """
    Set the model to distributed mode.
    Is not required in the case of single GPU training.
    """
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        #  find_unused_parameters=True,
        output_device=local_rank
        )
    return net

def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AverageMeter('loss')
    loss_pre_meter = AverageMeter('loss_prem')
    loss_aux_meters = [AverageMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters