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

from train.semantic_segmentation.loss_functions.ohem_ce_loss import OhemCELoss
from train.lr_scheduler_2 import WarmupPolyLrScheduler
from eval.utils import AverageMeter, TimeMeter
from utilities.print_utils import *

## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')