import cv2
import numpy as np
from PIL import Image
import os
import glob
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

UNDEFINED = 255
# OVER_RATIO = 1.5
DEBUG = False
THRESHOLD = 0

idToClassMap = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
                7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 255: 'undefined'}

cityscapesIdToClassMap = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
                7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 255: 'undefined'}

def create_pixel_region_map(mask, cnts):
    """
    register each pixel to the region

    :parameters
    ----------
    cnts: contours find by opencv

    mask: numpy array
        Height and width of mask as (height, width).

    :returns
    pixel_region_map: dictionary

    """
    pixel_object_map = dict()
    for i, cnt in enumerate(cnts):
        pts_xy = find_object_pixels(mask, cnt)

        for pt in pts_xy:
            pixel_cor = pt
            pixel_object_map[pixel_cor] = i

    return pixel_object_map

def find_contours(mask, class_val):
    """
    generate for a given class

    Parameters
    ----------
    mask: numpy array
        Height and width of mask as (height, width).
    class_val: int
        integer represents a class

    Returns
    -------
    binary mask: numpy array
    """
    # binary mask gt for the given class
    class_mask = np.zeros_like(mask, dtype=np.uint8)
    class_mask[mask == class_val] = 1

    # find all contours represent the class
    # contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def find_object_pixels(mask, cnt):
    """
    generate for a given class

    Parameters
    ----------
    mask: numpy array
        Height and width of mask as (height, width).
    cnt: one contour from cv.findcontour()

    Returns
    -------
    list of points (x, y): numpy array
    """
    # create an empty mask to mask the object
    single_object_mask = np.zeros_like(mask)
    # find all the pixel coordinates for the object
    cv2.drawContours(image=single_object_mask, contours=[cnt], contourIdx=-1, color=255, thickness=-1)

    pts = np.where(single_object_mask == 255)
    pts_xy = set(zip(pts[0], pts[1]))
    return pts_xy

def segmentation_score_Persello(gt, pred, class_val):
    """
    compute the score as proposed Persello

    :parameters
    ----------
    gt: numpy array
        Height and width of gt as (height, width).
    pred: numpy array
        Height and width of pred as (height, width).
    class_val: int
        integer represents a class

    :returns
    score: float

    """
    gt_contours = find_contours(gt, class_val)
    pred_contours = find_contours(pred, class_val)

    pred_pixel_object_map = create_pixel_region_map(pred, pred_contours)

    OS = []
    US = []

    for cnt in gt_contours:
        # all the points in this ground-truth region
        pts_gt = find_object_pixels(gt, cnt)

        pred_objects_overlap = dict()  # register all the predictions touched by this ground truth region

        # compare each points in this ground truth region to the prediction
        for pt_gt in pts_gt:
            pixel_cor = pt_gt

            # count how many pred region in this gt region
            if pixel_cor in pred_pixel_object_map:  # check if this pixel is also in one of the predicted region
                pred_object_id = pred_pixel_object_map[pixel_cor]
                if pred_object_id not in pred_objects_overlap:
                    pred_objects_overlap[pred_object_id] = 0
                else:
                    pred_objects_overlap[pred_object_id] = pred_objects_overlap[pred_object_id] + 1

        # number of prediction touched by this gt
        num_of_preds = len(pred_objects_overlap)

        if num_of_preds >= 1:
        # find the one with the largest overlap
            largest_overlapping_pred = max(pred_objects_overlap, key=pred_objects_overlap.get)

            OSi = 1 - pred_objects_overlap[largest_overlapping_pred]/len(pts_gt)
            OS.append(OSi)

            pts_pred = set([key for key, value in pred_pixel_object_map.items() if value == largest_overlapping_pred])
            USi = 1 - pred_objects_overlap[largest_overlapping_pred]/len(pts_pred)
            US.append(USi)

    return np.average(OS), np.average(US)