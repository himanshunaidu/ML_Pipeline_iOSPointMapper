import os
import torch
import numpy as np
from enum import Enum

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return self.summary()

    def summary(self):
        fmtstr = "AverageMeter Summary:"
        fmtstr += "\nValue: {val:.3f}"
        fmtstr += "\nAverage: {avg:.3f}"
        fmtstr += "\nSum: {sum:.3f}"
        fmtstr += "\nCount: {count:.3f}"
        
        return fmtstr.format(**self.__dict__)

if __name__=="__main__":
    # Test the AverageMeter class
    avg_meter = AverageMeter()
    avg_meter.update(10)
    avg_meter.update(20)
    print(avg_meter)