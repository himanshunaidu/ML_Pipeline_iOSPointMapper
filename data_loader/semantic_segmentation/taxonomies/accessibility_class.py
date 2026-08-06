from dataclasses import dataclass
from enum import Enum

import os
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass, ColorClass

class AccessibilityClass(Enum):
    ROAD = ProjectClass(key="road", name="Road")
    SIDEWALK = ProjectClass(key="sidewalk", name="Sidewalk")
    BUILDING = ProjectClass(key="building", name="Building")
    POLE = ProjectClass(key="pole", name="Pole")
    TRAFFIC_LIGHT = ProjectClass(key="traffic_light", name="Traffic Light")
    TRAFFIC_SIGN = ProjectClass(key="traffic_sign", name="Traffic Sign")
    CURB = ProjectClass(key="curb", name="Curb")
    CURB_RAMP = ProjectClass(key="curb_ramp", name="Curb Ramp")
    VEGETATION = ProjectClass(key="vegetation", name="Vegetation")
    TERRAIN = ProjectClass(key="terrain", name="Terrain")
    WALL = ProjectClass(key="wall", name="Wall")
    FENCE = ProjectClass(key="fence", name="Fence")
    STATIC = ProjectClass(key="static", name="Static Object")
    DYNAMIC = ProjectClass(key="dynamic", name="Dynamic Object")
    BACKGROUND = ProjectClass(key="background", name="Background")
    
class AccessibilityColorClassDefault(Enum):
    """
    Matches the cityscapes color scheme where possible.    
    Matches the custom ios_point_mapper color scheme where possible.
    """
    ROAD = ColorClass(project_class=AccessibilityClass.ROAD, rgb=np.array([128, 64, 128]))
    SIDEWALK = ColorClass(project_class=AccessibilityClass.SIDEWALK, rgb=np.array([244, 35, 232]))
    BUILDING = ColorClass(project_class=AccessibilityClass.BUILDING, rgb=np.array([70, 70, 70]))
    WALL = ColorClass(project_class=AccessibilityClass.WALL, rgb=np.array([102, 102, 156]))
    FENCE = ColorClass(project_class=AccessibilityClass.FENCE, rgb=np.array([190, 153, 153]))
    POLE = ColorClass(project_class=AccessibilityClass.POLE, rgb=np.array([153, 153, 153]))
    TRAFFIC_LIGHT = ColorClass(project_class=AccessibilityClass.TRAFFIC_LIGHT, rgb=np.array([250, 170, 30]))
    TRAFFIC_SIGN = ColorClass(project_class=AccessibilityClass.TRAFFIC_SIGN, rgb=np.array([220, 220, 0]))
    VEGETATION = ColorClass(project_class=AccessibilityClass.VEGETATION, rgb=np.array([107, 142, 35]))
    TERRAIN = ColorClass(project_class=AccessibilityClass.TERRAIN, rgb=np.array([152, 251, 152]))
    STATIC = ColorClass(project_class=AccessibilityClass.STATIC, rgb=np.array([70, 130, 180]))
    DYNAMIC = ColorClass(project_class=AccessibilityClass.DYNAMIC, rgb=np.array([220, 20, 60]))
    BACKGROUND = ColorClass(project_class=AccessibilityClass.BACKGROUND, rgb=np.array([0, 0, 0]))
    CURB = ColorClass(project_class=AccessibilityClass.CURB, rgb=np.array([204, 51, 102]))
    CURB_RAMP = ColorClass(project_class=AccessibilityClass.CURB_RAMP, rgb=np.array([255, 255, 0]))
    

ACCESSIBILITY_11_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN,
    AccessibilityClass.STATIC,
    AccessibilityClass.DYNAMIC
)

ACCESSIBILITY_12_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.CURB_RAMP,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN,
    AccessibilityClass.STATIC,
    AccessibilityClass.DYNAMIC
)

ACCESSIBILITY_13_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.CURB,
    AccessibilityClass.CURB_RAMP,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN,
    AccessibilityClass.STATIC,
    AccessibilityClass.DYNAMIC
)

ACCESSIBILITY_3_CURB_RAMP_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.CURB_RAMP
)


# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]
ACCESSIBILITY_CITYSCAPES_CLASSES = (
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.WALL,
    AccessibilityClass.FENCE,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN
)