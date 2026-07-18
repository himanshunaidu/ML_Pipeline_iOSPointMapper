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