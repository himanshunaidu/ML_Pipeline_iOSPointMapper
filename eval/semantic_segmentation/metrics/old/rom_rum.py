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

def rom_rum(gt, pred, class_val):
    gt_contours = find_contours(gt, class_val)
    pred_contours = find_contours(pred, class_val)

    G_Ib = len(gt_contours)
    S_Ib = len(pred_contours)

    # generate pixel-object region map for the ground truth
    gt_pixel_object_map = create_pixel_region_map(gt, gt_contours)
    gt_objects = set(gt_pixel_object_map.values())

    # generate pixel-object region map for the prediction
    pred_pixel_object_map = create_pixel_region_map(pred, pred_contours)
    pred_objects = set(pred_pixel_object_map.values())

    # Compute ROM
    S_IO = set()
    G_IO = set()

    AO = 0
    for cnt in gt_contours:
        pts_gt = find_object_pixels(gt, cnt)

        pred_objects_overlap = set()  # register all the prediction region touched by this gt

        for pt in pts_gt:
            pixel_cor = pt

            # register all the prediction region touched by this gt
            if pixel_cor in pred_pixel_object_map:
                if pred_pixel_object_map[pixel_cor] not in pred_objects_overlap:
                    pred_objects_overlap.add(pred_pixel_object_map[pixel_cor])

        num_of_preds = len(pred_objects_overlap)

        if num_of_preds > 1:
            # all preds touched by this gt contributing to over-segmentation
            # S_IO.add(x for x in pred_objects_overlap if x not in S_IO)
            S_IO = S_IO.union(pred_objects_overlap)
            G_IO.add(gt_pixel_object_map[pixel_cor])

        # print(x for x in pred_objects_overlap)
        AO += max(num_of_preds-1, 0)

    AO = AO/(G_Ib+0.00001)

    ROR = len(S_IO)*len(G_IO)/(G_Ib*S_Ib+0.001)
    ROM = ROR*np.exp(-1/(AO+0.001))

    # Compute RUM
    S_IU = set()
    G_IU = set()

    AU = 0
    for cnt in pred_contours:
        pts_pred = find_object_pixels(pred, cnt)

        gt_objects_overlap = set()  # register all the gt region touched by this prediction

        for pt in pts_pred:
            pixel_cor = pt

            # register all the gt region touched by this prediction
            if pixel_cor in gt_pixel_object_map:
                if gt_pixel_object_map[pixel_cor] not in gt_objects_overlap:
                    gt_objects_overlap.add(gt_pixel_object_map[pixel_cor])

        num_of_gts = len(gt_objects_overlap)

        if num_of_gts > 1:
            # all gts touched by this pred involves in under-segmentation
            G_IU = G_IU.union(gt_objects_overlap)
            S_IU.add(pred_pixel_object_map[pixel_cor])

        # print(x for x in pred_objects_overlap)
        AU += max(num_of_gts-1, 0)

    AU = AU/(S_Ib+0.00001)

    RUR = len(S_IU)*len(G_IU)/(G_Ib*S_Ib+0.001)
    RUM = RUR*np.exp(-1/(AU+0.001))

    return ROM, RUM