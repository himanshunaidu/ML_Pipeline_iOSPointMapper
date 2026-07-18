from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass
from data_loader.semantic_segmentation.taxonomies.accessibility_class import AccessibilityClass, ACCESSIBILITY_11_CLASSES, ACCESSIBILITY_12_CLASSES

class CityscapesClass(Enum):
    ROAD = SourceClass(raw_id=0, name='road')
    SIDEWALK = SourceClass(raw_id=1, name='sidewalk')
    BUILDING = SourceClass(raw_id=2, name='building')
    WALL = SourceClass(raw_id=3, name='wall')
    FENCE = SourceClass(raw_id=4, name='fence')
    POLE = SourceClass(raw_id=5, name='pole')
    TRAFFIC_LIGHT = SourceClass(raw_id=6, name='traffic light')
    TRAFFIC_SIGN = SourceClass(raw_id=7, name='traffic sign')
    VEGETATION = SourceClass(raw_id=8, name='vegetation')
    TERRAIN = SourceClass(raw_id=9, name='terrain')
    SKY = SourceClass(raw_id=10, name='sky')
    PERSON = SourceClass(raw_id=11, name='person')
    RIDER = SourceClass(raw_id=12, name='rider')
    CAR = SourceClass(raw_id=13, name='car')
    TRUCK = SourceClass(raw_id=14, name='truck')
    BUS = SourceClass(raw_id=15, name='bus')
    TRAIN = SourceClass(raw_id=16, name='train')
    MOTORCYCLE = SourceClass(raw_id=17, name='motorcycle')
    BICYCLE = SourceClass(raw_id=18, name='bicycle')
    BACKGROUND = SourceClass(raw_id=19, name='background')

