from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass

class iOSPointMapperClass(Enum):
    BACKGROUND = SourceClass(raw_id=0, name='background')
    BICYCLE = SourceClass(raw_id=1, name='bicycle')
    BIKE_RACK = SourceClass(raw_id=2, name='bike rack')
    BRIDGE = SourceClass(raw_id=3, name='bridge')
    BUILDING = SourceClass(raw_id=4, name='building')
    BUS = SourceClass(raw_id=5, name='bus')
    CAR = SourceClass(raw_id=6, name='car')
    CROSSWALK = SourceClass(raw_id=7, name='crosswalk')
    CURB = SourceClass(raw_id=8, name='curb')
    CURB_RAMP = SourceClass(raw_id=9, name='curb ramp')
    DYNAMIC = SourceClass(raw_id=10, name='dynamic')
    FENCE = SourceClass(raw_id=11, name='fence')
    GROUND = SourceClass(raw_id=12, name='ground')
    GUARD_RAIL = SourceClass(raw_id=13, name='guard rail')
    LICENSE_PLATE = SourceClass(raw_id=14, name='license plate')
    MOTORCYCLE = SourceClass(raw_id=15, name='motorcycle')
    PARKING = SourceClass(raw_id=16, name='parking')
    PERSON = SourceClass(raw_id=17, name='person')
    POLE = SourceClass(raw_id=18, name='pole')
    RAIL_TRACK = SourceClass(raw_id=19, name='rail track')
    RIDER = SourceClass(raw_id=20, name='rider')
    ROAD = SourceClass(raw_id=21, name='road')
    SIDEWALK = SourceClass(raw_id=22, name='sidewalk')
    SKY = SourceClass(raw_id=23, name='sky')
    STATIC = SourceClass(raw_id=24, name='static')
    TACTILE_PAVING = SourceClass(raw_id=25, name='tactile paving')
    TERRAIN = SourceClass(raw_id=26, name='terrain')
    TRAFFIC_LIGHT = SourceClass(raw_id=27, name='traffic light')
    TRAFFIC_SIGN = SourceClass(raw_id=28, name='traffic sign')
    TRAIN = SourceClass(raw_id=29, name='train')
    TRUCK = SourceClass(raw_id=30, name='truck')
    TUNNEL = SourceClass(raw_id=31, name='tunnel')
    VEGETATION = SourceClass(raw_id=32, name='vegetation')
    WALL = SourceClass(raw_id=33, name='wall')
    