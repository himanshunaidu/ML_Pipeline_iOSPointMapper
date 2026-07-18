from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass

class MapillaryVistasClass(Enum):
    # Construction types
    CONSTRUCTION_BARRIER_AMBIGUOUS = SourceClass(raw_id=2, name='construction--barrier--ambiguous')
    CONSTRUCTION_BARRIER_CONCRETE_BLOCK = SourceClass(raw_id=3, name='construction--barrier--concrete-block')
    CONSTRUCTION_BARRIER_CURB = SourceClass(raw_id=4, name='construction--barrier--curb')
    CONSTRUCTION_BARRIER_FENCE = SourceClass(raw_id=5, name='construction--barrier--fence')
    CONSTRUCTION_BARRIER_GUARD_RAIL = SourceClass(raw_id=6, name='construction--barrier--guard-rail')
    CONSTRUCTION_BARRIER_OTHER_BARRIER = SourceClass(raw_id=7, name='construction--barrier--other-barrier')
    CONSTRUCTION_BARRIER_ROAD_MEDIAN = SourceClass(raw_id=8, name='construction--barrier--road-median')
    CONSTRUCTION_BARRIER_ROAD_SIDE = SourceClass(raw_id=9, name='construction--barrier--road-side')
    CONSTRUCTION_BARRIER_SEPARATOR = SourceClass(raw_id=10, name='construction--barrier--separator')
    # CONSTRUCTION_BARRIER_TEMPORARY = SourceClass(raw_id=11, name='construction--barrier--temporary')
    CONSTRUCTION_BARRIER_WALL = SourceClass(raw_id=12, name='construction--barrier--wall')
    CONSTRUCTION_FLAT_BIKE_LANE = SourceClass(raw_id=13, name='construction--flat--bike-lane')
    CONSTRUCTION_FLAT_CROSSWALK_PLAIN = SourceClass(raw_id=14, name='construction--flat--crosswalk-plain')
    CONSTRUCTION_FLAT_CURB_CUT = SourceClass(raw_id=15, name='construction--flat--curb-cut')
    CONSTRUCTION_FLAT_DRIVEWAY = SourceClass(raw_id=16, name='construction--flat--driveway')
    ## Parking types
    CONSTRUCTION_FLAT_PARKING = SourceClass(raw_id=17, name='construction--flat--parking')
    CONSTRUCTION_FLAT_PARKING_AISLE = SourceClass(raw_id=18, name='construction--flat--parking-aisle')
    ## Sidewalk types
    CONSTRUCTION_FLAT_PEDESTRIAN_AREA = SourceClass(raw_id=19, name='construction--flat--pedestrian-area')
    CONSTRUCTION_FLAT_SIDEWALK = SourceClass(raw_id=24, name='construction--flat--sidewalk')
    
    CONSTRUCTION_FLAT_RAIL_TRACK = SourceClass(raw_id=20, name='construction--flat--rail-track')
    ## Road types
    CONSTRUCTION_FLAT_ROAD = SourceClass(raw_id=21, name='construction--flat--road')
    CONSTRUCTION_FLAT_SERVICE_LANE = SourceClass(raw_id=23, name='construction--flat--service-lane')
    CONSTRUCTION_FLAT_ROAD_SHOULDER = SourceClass(raw_id=22, name='construction--flat--road-shoulder')
    CONSTRUCTION_FLAT_TRAFFIC_ISLAND = SourceClass(raw_id=25, name='construction--flat--traffic-island')
    
    # Other vehicle types
    OBJECT_VEHICLE_CARAVAN = SourceClass(raw_id=109, name='object--vehicle--caravan')
    OBJECT_VEHICLE_ON_RAILS = SourceClass(raw_id=111, name='object--vehicle--on-rails')
    OBJECT_VEHICLE_OTHER_VEHICLE = SourceClass(raw_id=112, name='object--vehicle--other-vehicle')
    OBJECT_VEHICLE_TRAILER = SourceClass(raw_id=113, name='object--vehicle--trailer')
    OBJECT_VEHICLE_VEHICLE_GROUP = SourceClass(raw_id=115, name='object--vehicle--vehicle-group')
    OBJECT_VEHICLE_WHEELED_SLOW = SourceClass(raw_id=116, name='object--vehicle--wheeled-slow')
    
    CONSTRUCTION_STRUCTURE_BRIDGE = SourceClass(raw_id=26, name='construction--structure--bridge')
    # Building types
    CONSTRUCTION_STRUCTURE_BUILDING = SourceClass(raw_id=27, name='construction--structure--building')
    CONSTRUCTION_STRUCTURE_GARAGE = SourceClass(raw_id=28, name='construction--structure--garage')
    
    CONSTRUCTION_STRUCTURE_TUNNEL = SourceClass(raw_id=29, name='construction--structure--tunnel')
    ## Person types
    HUMAN_PERSON_INDIVIDUAL = SourceClass(raw_id=30, name='human--person--individual')
    HUMAN_PERSON_PERSON_GROUP = SourceClass(raw_id=31, name='human--person--person-group')
    HUMAN_RIDER_BICYCLIST = SourceClass(raw_id=32, name='human--rider--bicyclist')
    HUMAN_RIDER_MOTORCYCLIST = SourceClass(raw_id=33, name='human--rider--motorcyclist')
    HUMAN_RIDER_OTHER_RIDER = SourceClass(raw_id=34, name='human--rider--other-rider')
    
    ## Crosswalk types
    MARKING_DISCRETE_CROSSWALK_ZEBRA = SourceClass(raw_id=45, name='marking--discrete--crosswalk-zebra')
    MARKING_ONLY_DISCRETE_CROSSWALK_ZEBRA = SourceClass(raw_id=56, name='marking-only--discrete--crosswalk-zebra')
    
    ## Terrain types
    NATURE_SAND = SourceClass(raw_id=60, name='nature--sand')
    NATURE_SNOW = SourceClass(raw_id=62, name='nature--snow')
    NATURE_TERRAIN = SourceClass(raw_id=63, name='nature--terrain')
    
    NATURE_SKY = SourceClass(raw_id=61, name='nature--sky')
    
    ## Vegetation types
    NATURE_VEGETATION = SourceClass(raw_id=64, name='nature--vegetation')
    
    OBJECT_BENCH = SourceClass(raw_id=67, name='object--bench')
    OBJECT_BIKE_RACK = SourceClass(raw_id=68, name='object--bike-rack')
    OBJECT_CATCH_BASIN = SourceClass(raw_id=69, name='object--catch-basin')
    OBJECT_FIRE_HYDRANT = SourceClass(raw_id=71, name='object--fire-hydrant')
    OBJECT_JUNCTION_BOX = SourceClass(raw_id=72, name='object--junction-box')
    OBJECT_MAILBOX = SourceClass(raw_id=73, name='object--mailbox')
    OBJECT_PARKING_METER = SourceClass(raw_id=75, name='object--parking-meter')
    OBJECT_PHONE_BOOTH = SourceClass(raw_id=76, name='object--phone-booth')
    # OBJECT_POTHOLE = SourceClass(raw_id=77, name='object--pothole')
    
    ## Signage types
    OBJECT_SIGN_ADVERTISEMENT = SourceClass(raw_id=78, name='object--sign--advertisement')
    OBJECT_SIGN_AMBIGUOUS = SourceClass(raw_id=79, name='object--sign--ambiguous')
    OBJECT_SIGN_BACK = SourceClass(raw_id=80, name='object--sign--back')
    OBJECT_SIGN_INFORMATION = SourceClass(raw_id=81, name='object--sign--information')
    OBJECT_SIGN_OTHER = SourceClass(raw_id=82, name='object--sign--other')
    OBJECT_SIGN_STORE = SourceClass(raw_id=83, name='object--sign--store')
    
    ## Pole types
    OBJECT_STREET_LIGHT = SourceClass(raw_id=84, name='object--street-light') # Considered as pole for now.
    OBJECT_SUPPORT_POLE = SourceClass(raw_id=85, name='object--support--pole')
    OBJECT_SUPPORT_POLE_GROUP = SourceClass(raw_id=86, name='object--support--pole-group')
    OBJECT_SUPPORT_TRAFFIC_SIGN_FRAME = SourceClass(raw_id=87, name='object--support--traffic-sign-frame')
    OBJECT_SUPPORT_UTILITY_POLE = SourceClass(raw_id=88, name='object--support--utility-pole')
    
    ## Traffic sign types
    OBJECT_TRAFFIC_SIGN_AMBIGUOUS = SourceClass(raw_id=96, name='object--traffic-sign--ambiguous')
    OBJECT_TRAFFIC_SIGN_BACK = SourceClass(raw_id=97, name='object--traffic-sign--back')
    OBJECT_TRAFFIC_SIGN_DIRECTION_BACK = SourceClass(raw_id=98, name='object--traffic-sign--direction-back')
    OBJECT_TRAFFIC_SIGN_DIRECTION_FRONT = SourceClass(raw_id=99, name='object--traffic-sign--direction-front')
    OBJECT_TRAFFIC_SIGN_FRONT = SourceClass(raw_id=100, name='object--traffic-sign--front')
    OBJECT_TRAFFIC_SIGN_INFORMATION_PARKING = SourceClass(raw_id=101, name='object--traffic-sign--information-parking')
    OBJECT_TRAFFIC_SIGN_TEMPORARY_BACK = SourceClass(raw_id=102, name='object--traffic-sign--temporary-back')
    OBJECT_TRAFFIC_SIGN_TEMPORARY_FRONT = SourceClass(raw_id=103, name='object--traffic-sign--temporary-front')
    
    ## Traffic light types
    OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE = SourceClass(raw_id=90, name='object--traffic-light--general-single')
    OBJECT_TRAFFIC_LIGHT_PEDESTRIANS = SourceClass(raw_id=91, name='object--traffic-light--pedestrians')
    OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT = SourceClass(raw_id=92, name='object--traffic-light--general-upright')
    OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL = SourceClass(raw_id=93, name='object--traffic-light--general-horizontal')
    OBJECT_TRAFFIC_LIGHT_CYCLISTS = SourceClass(raw_id=94, name='object--traffic-light--cyclists')
    OBJECT_TRAFFIC_LIGHT_OTHER = SourceClass(raw_id=95, name='object--traffic-light--other')
    
    OBJECT_TRASH_CAN = SourceClass(raw_id=104, name='object--trash-can')
    OBJECT_VEHICLE_BICYCLE = SourceClass(raw_id=105, name='object--vehicle--bicycle')
    OBJECT_VEHICLE_BOAT = SourceClass(raw_id=106, name='object--vehicle--boat')
    OBJECT_VEHICLE_BUS = SourceClass(raw_id=107, name='object--vehicle--bus')
    OBJECT_VEHICLE_CAR = SourceClass(raw_id=108, name='object--vehicle--car')
    OBJECT_VEHICLE_MOTORCYCLE = SourceClass(raw_id=110, name='object--vehicle--motorcycle')
    OBJECT_VEHICLE_TRUCK = SourceClass(raw_id=114, name='object--vehicle--truck')
    
    ## Lane marking types
    MARKING_CONTINUOUS_DASHED = SourceClass(raw_id=35, name='marking--continuous--dashed')
    MARKING_CONTINUOUS_SOLID = SourceClass(raw_id=36, name='marking--continuous--solid')
    MARKING_CONTINUOUS_ZIGZAG = SourceClass(raw_id=37, name='marking--continuous--zigzag')
    MARKING_DISCRETE_AMBIGUOUS = SourceClass(raw_id=38, name='marking--discrete--ambiguous')
    MARKING_DISCRETE_ARROW_LEFT = SourceClass(raw_id=39, name='marking--discrete--arrow--left')
    MARKING_DISCRETE_ARROW_OTHER = SourceClass(raw_id=40, name='marking--discrete--arrow--other')
    MARKING_DISCRETE_ARROW_RIGHT = SourceClass(raw_id=41, name='marking--discrete--arrow--right')
    MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_STRAIGHT = SourceClass(raw_id=42, name='marking--discrete--arrow--split-left-or-straight')
    MARKING_DISCRETE_ARROW_SPLIT_RIGHT_OR_STRAIGHT = SourceClass(raw_id=43, name='marking--discrete--arrow--split-right-or-straight')
    MARKING_DISCRETE_ARROW_STRAIGHT = SourceClass(raw_id=44, name='marking--discrete--arrow--straight')
    MARKING_DISCRETE_GIVE_WAY_ROW = SourceClass(raw_id=46, name='marking--discrete--give-way-row')
    MARKING_DISCRETE_GIVE_WAY_SINGLE = SourceClass(raw_id=47, name='marking--discrete--give-way-single')
    MARKING_DISCRETE_HATCHED_CHEVRON = SourceClass(raw_id=48, name='marking--discrete--hatched--chevron')
    MARKING_DISCRETE_HATCHED_DIAGONAL = SourceClass(raw_id=49, name='marking--discrete--hatched--diagonal')
    MARKING_DISCRETE_OTHER_MARKING = SourceClass(raw_id=50, name='marking--discrete--other-marking')
    MARKING_DISCRETE_STOP_LINE = SourceClass(raw_id=51, name='marking--discrete--stop-line')
    MARKING_DISCRETE_SYMBOL_BICYCLE = SourceClass(raw_id=52, name='marking--discrete--symbol--bicycle')
    MARKING_DISCRETE_SYMBOL_OTHER = SourceClass(raw_id=53, name='marking--discrete--symbol--other')
    MARKING_DISCRETE_TEXT = SourceClass(raw_id=54, name='marking--discrete--text')
    MARKING_ONLY_CONTINUOUS_DASHED = SourceClass(raw_id=55, name='marking-only--continuous--dashed')
    MARKING_ONLY_DISCRETE_OTHER_MARKING = SourceClass(raw_id=57, name='marking-only--discrete--other-marking')
    MARKING_ONLY_DISCRETE_TEXT = SourceClass(raw_id=58, name='marking-only--discrete--text')
    
    
