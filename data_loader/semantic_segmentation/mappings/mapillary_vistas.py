from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass
from data_loader.semantic_segmentation.taxonomies.mapillary_vistas import MapillaryVistasClass
from data_loader.semantic_segmentation.taxonomies.accessibility_class import AccessibilityClass, ACCESSIBILITY_11_CLASSES, ACCESSIBILITY_12_CLASSES, ACCESSIBILITY_13_CLASSES, ACCESSIBILITY_3_CURB_RAMP_CLASSES
from data_loader.semantic_segmentation.mappings.builder import MappingSpec, MappingBuilder

MAPILLARY_VISTAS_ACCESSIBILITY_11: MappingSpec[MapillaryVistasClass, AccessibilityClass] = (
    MappingBuilder(
        name="mapillary_vistas_accessibility_11",
        source_enum=MapillaryVistasClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_11_CLASSES,
    )
    .map(MapillaryVistasClass.NATURE_SKY, AccessibilityClass.BACKGROUND)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_ROAD, MapillaryVistasClass.CONSTRUCTION_FLAT_ROAD_SHOULDER,
        ## markings
        MapillaryVistasClass.MARKING_CONTINUOUS_DASHED, MapillaryVistasClass.MARKING_CONTINUOUS_SOLID, MapillaryVistasClass.MARKING_CONTINUOUS_ZIGZAG,
        MapillaryVistasClass.MARKING_DISCRETE_AMBIGUOUS, MapillaryVistasClass.MARKING_DISCRETE_ARROW_LEFT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_OTHER, MapillaryVistasClass.MARKING_DISCRETE_ARROW_RIGHT,
        MapillaryVistasClass.MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_STRAIGHT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_SPLIT_RIGHT_OR_STRAIGHT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_STRAIGHT,
        MapillaryVistasClass.MARKING_DISCRETE_GIVE_WAY_ROW, MapillaryVistasClass.MARKING_DISCRETE_GIVE_WAY_SINGLE, MapillaryVistasClass.MARKING_DISCRETE_HATCHED_CHEVRON, 
        MapillaryVistasClass.MARKING_DISCRETE_HATCHED_DIAGONAL, MapillaryVistasClass.MARKING_DISCRETE_OTHER_MARKING, MapillaryVistasClass.MARKING_DISCRETE_STOP_LINE, MapillaryVistasClass.MARKING_DISCRETE_SYMBOL_BICYCLE, 
        MapillaryVistasClass.MARKING_DISCRETE_SYMBOL_OTHER, MapillaryVistasClass.MARKING_DISCRETE_TEXT, 
        MapillaryVistasClass.MARKING_ONLY_CONTINUOUS_DASHED, MapillaryVistasClass.MARKING_ONLY_DISCRETE_OTHER_MARKING, MapillaryVistasClass.MARKING_ONLY_DISCRETE_TEXT,
        ## curb
        MapillaryVistasClass.CONSTRUCTION_BARRIER_CURB
    ], AccessibilityClass.ROAD)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_PEDESTRIAN_AREA, MapillaryVistasClass.CONSTRUCTION_FLAT_SIDEWALK,
        ## curb cut/ramp
        MapillaryVistasClass.CONSTRUCTION_FLAT_CURB_CUT
    ], AccessibilityClass.SIDEWALK)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_STRUCTURE_BUILDING, MapillaryVistasClass.CONSTRUCTION_STRUCTURE_GARAGE
    ], AccessibilityClass.BUILDING)
    .map_many([
        MapillaryVistasClass.OBJECT_STREET_LIGHT, ### street light
        MapillaryVistasClass.OBJECT_SUPPORT_POLE, MapillaryVistasClass.OBJECT_SUPPORT_POLE_GROUP,
        MapillaryVistasClass.OBJECT_SUPPORT_TRAFFIC_SIGN_FRAME, MapillaryVistasClass.OBJECT_SUPPORT_UTILITY_POLE
    ], AccessibilityClass.POLE)
    .map_many([
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_PEDESTRIANS,
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL,
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_CYCLISTS, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_OTHER
    ], AccessibilityClass.TRAFFIC_LIGHT)
    .map_many([
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_AMBIGUOUS, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_BACK,
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_DIRECTION_BACK, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_DIRECTION_FRONT,
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_FRONT, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_INFORMATION_PARKING, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_TEMPORARY_BACK, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_TEMPORARY_FRONT
    ], AccessibilityClass.TRAFFIC_SIGN)
    .map_many([
        MapillaryVistasClass.NATURE_VEGETATION
    ], AccessibilityClass.VEGETATION)
    .map_many([
        MapillaryVistasClass.NATURE_SAND, MapillaryVistasClass.NATURE_SNOW, MapillaryVistasClass.NATURE_TERRAIN
    ], AccessibilityClass.TERRAIN)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_BARRIER_AMBIGUOUS, MapillaryVistasClass.CONSTRUCTION_BARRIER_CONCRETE_BLOCK, MapillaryVistasClass.CONSTRUCTION_BARRIER_FENCE,
        MapillaryVistasClass.CONSTRUCTION_BARRIER_GUARD_RAIL, MapillaryVistasClass.CONSTRUCTION_BARRIER_OTHER_BARRIER, MapillaryVistasClass.CONSTRUCTION_BARRIER_ROAD_MEDIAN, 
        MapillaryVistasClass.CONSTRUCTION_BARRIER_ROAD_SIDE, MapillaryVistasClass.CONSTRUCTION_BARRIER_SEPARATOR, MapillaryVistasClass.CONSTRUCTION_BARRIER_WALL,
        MapillaryVistasClass.OBJECT_BIKE_RACK
    ], AccessibilityClass.STATIC)
    .map_many([
        MapillaryVistasClass.OBJECT_TRASH_CAN, MapillaryVistasClass.OBJECT_VEHICLE_BICYCLE, MapillaryVistasClass.OBJECT_VEHICLE_BOAT, MapillaryVistasClass.OBJECT_VEHICLE_BUS,
        MapillaryVistasClass.OBJECT_VEHICLE_CAR, MapillaryVistasClass.OBJECT_VEHICLE_MOTORCYCLE, MapillaryVistasClass.OBJECT_VEHICLE_TRUCK, MapillaryVistasClass.OBJECT_VEHICLE_CARAVAN, 
        MapillaryVistasClass.OBJECT_VEHICLE_ON_RAILS, MapillaryVistasClass.OBJECT_VEHICLE_OTHER_VEHICLE, MapillaryVistasClass.OBJECT_VEHICLE_TRAILER, MapillaryVistasClass.OBJECT_VEHICLE_VEHICLE_GROUP,
        MapillaryVistasClass.OBJECT_VEHICLE_WHEELED_SLOW, 
        MapillaryVistasClass.HUMAN_PERSON_INDIVIDUAL, MapillaryVistasClass.HUMAN_PERSON_PERSON_GROUP, MapillaryVistasClass.HUMAN_RIDER_BICYCLIST, MapillaryVistasClass.HUMAN_RIDER_MOTORCYCLIST, 
        MapillaryVistasClass.HUMAN_RIDER_OTHER_RIDER,
        MapillaryVistasClass.OBJECT_BENCH, MapillaryVistasClass.OBJECT_FIRE_HYDRANT, MapillaryVistasClass.OBJECT_JUNCTION_BOX, 
        MapillaryVistasClass.OBJECT_MAILBOX, MapillaryVistasClass.OBJECT_PARKING_METER, MapillaryVistasClass.OBJECT_PHONE_BOOTH,
        MapillaryVistasClass.OBJECT_SIGN_ADVERTISEMENT, MapillaryVistasClass.OBJECT_SIGN_AMBIGUOUS, MapillaryVistasClass.OBJECT_SIGN_BACK, 
        MapillaryVistasClass.OBJECT_SIGN_INFORMATION, MapillaryVistasClass.OBJECT_SIGN_OTHER, MapillaryVistasClass.OBJECT_SIGN_STORE
    ], AccessibilityClass.DYNAMIC)
    .build()
)

MAPILLARY_VISTAS_ACCESSIBILITY_12: MappingSpec[MapillaryVistasClass, AccessibilityClass] = (
    MappingBuilder(
        name="mapillary_vistas_accessibility_12",
        source_enum=MapillaryVistasClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_12_CLASSES,
    )
    .map_many([
        MapillaryVistasClass.NATURE_SKY,
        MapillaryVistasClass.CONSTRUCTION_BARRIER_CURB
    ], AccessibilityClass.BACKGROUND)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_ROAD, MapillaryVistasClass.CONSTRUCTION_FLAT_ROAD_SHOULDER,
        ## markings
        MapillaryVistasClass.MARKING_CONTINUOUS_DASHED, MapillaryVistasClass.MARKING_CONTINUOUS_SOLID, MapillaryVistasClass.MARKING_CONTINUOUS_ZIGZAG,
        MapillaryVistasClass.MARKING_DISCRETE_AMBIGUOUS, MapillaryVistasClass.MARKING_DISCRETE_ARROW_LEFT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_OTHER, MapillaryVistasClass.MARKING_DISCRETE_ARROW_RIGHT,
        MapillaryVistasClass.MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_STRAIGHT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_SPLIT_RIGHT_OR_STRAIGHT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_STRAIGHT,
        MapillaryVistasClass.MARKING_DISCRETE_GIVE_WAY_ROW, MapillaryVistasClass.MARKING_DISCRETE_GIVE_WAY_SINGLE, MapillaryVistasClass.MARKING_DISCRETE_HATCHED_CHEVRON, 
        MapillaryVistasClass.MARKING_DISCRETE_HATCHED_DIAGONAL, MapillaryVistasClass.MARKING_DISCRETE_OTHER_MARKING, MapillaryVistasClass.MARKING_DISCRETE_STOP_LINE, MapillaryVistasClass.MARKING_DISCRETE_SYMBOL_BICYCLE, 
        MapillaryVistasClass.MARKING_DISCRETE_SYMBOL_OTHER, MapillaryVistasClass.MARKING_DISCRETE_TEXT, 
        MapillaryVistasClass.MARKING_ONLY_CONTINUOUS_DASHED, MapillaryVistasClass.MARKING_ONLY_DISCRETE_OTHER_MARKING, MapillaryVistasClass.MARKING_ONLY_DISCRETE_TEXT,
        ## curb
        # MapillaryVistasClass.CONSTRUCTION_BARRIER_CURB
    ], AccessibilityClass.ROAD)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_PEDESTRIAN_AREA, MapillaryVistasClass.CONSTRUCTION_FLAT_SIDEWALK,
    ], AccessibilityClass.SIDEWALK)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_STRUCTURE_BUILDING, MapillaryVistasClass.CONSTRUCTION_STRUCTURE_GARAGE
    ], AccessibilityClass.BUILDING)
    .map_many([
        MapillaryVistasClass.OBJECT_STREET_LIGHT, ### street light
        MapillaryVistasClass.OBJECT_SUPPORT_POLE, MapillaryVistasClass.OBJECT_SUPPORT_POLE_GROUP,
        MapillaryVistasClass.OBJECT_SUPPORT_TRAFFIC_SIGN_FRAME, MapillaryVistasClass.OBJECT_SUPPORT_UTILITY_POLE
    ], AccessibilityClass.POLE)
    .map_many([
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_PEDESTRIANS,
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL,
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_CYCLISTS, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_OTHER
    ], AccessibilityClass.TRAFFIC_LIGHT)
    .map_many([
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_AMBIGUOUS, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_BACK,
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_DIRECTION_BACK, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_DIRECTION_FRONT,
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_FRONT, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_INFORMATION_PARKING, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_TEMPORARY_BACK, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_TEMPORARY_FRONT
    ], AccessibilityClass.TRAFFIC_SIGN)
    .map(MapillaryVistasClass.CONSTRUCTION_FLAT_CURB_CUT, AccessibilityClass.CURB_RAMP)
    .map_many([
        MapillaryVistasClass.NATURE_VEGETATION
    ], AccessibilityClass.VEGETATION)
    .map_many([
        MapillaryVistasClass.NATURE_SAND, MapillaryVistasClass.NATURE_SNOW, MapillaryVistasClass.NATURE_TERRAIN
    ], AccessibilityClass.TERRAIN)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_BARRIER_AMBIGUOUS, MapillaryVistasClass.CONSTRUCTION_BARRIER_CONCRETE_BLOCK, MapillaryVistasClass.CONSTRUCTION_BARRIER_FENCE,
        MapillaryVistasClass.CONSTRUCTION_BARRIER_GUARD_RAIL, MapillaryVistasClass.CONSTRUCTION_BARRIER_OTHER_BARRIER, MapillaryVistasClass.CONSTRUCTION_BARRIER_ROAD_MEDIAN, 
        MapillaryVistasClass.CONSTRUCTION_BARRIER_ROAD_SIDE, MapillaryVistasClass.CONSTRUCTION_BARRIER_SEPARATOR, MapillaryVistasClass.CONSTRUCTION_BARRIER_WALL,
        MapillaryVistasClass.OBJECT_BIKE_RACK
    ], AccessibilityClass.STATIC)
    .map_many([
        MapillaryVistasClass.OBJECT_TRASH_CAN, MapillaryVistasClass.OBJECT_VEHICLE_BICYCLE, MapillaryVistasClass.OBJECT_VEHICLE_BOAT, MapillaryVistasClass.OBJECT_VEHICLE_BUS,
        MapillaryVistasClass.OBJECT_VEHICLE_CAR, MapillaryVistasClass.OBJECT_VEHICLE_MOTORCYCLE, MapillaryVistasClass.OBJECT_VEHICLE_TRUCK, MapillaryVistasClass.OBJECT_VEHICLE_CARAVAN, 
        MapillaryVistasClass.OBJECT_VEHICLE_ON_RAILS, MapillaryVistasClass.OBJECT_VEHICLE_OTHER_VEHICLE, MapillaryVistasClass.OBJECT_VEHICLE_TRAILER, MapillaryVistasClass.OBJECT_VEHICLE_VEHICLE_GROUP,
        MapillaryVistasClass.OBJECT_VEHICLE_WHEELED_SLOW, 
        MapillaryVistasClass.HUMAN_PERSON_INDIVIDUAL, MapillaryVistasClass.HUMAN_PERSON_PERSON_GROUP, MapillaryVistasClass.HUMAN_RIDER_BICYCLIST, MapillaryVistasClass.HUMAN_RIDER_MOTORCYCLIST, 
        MapillaryVistasClass.HUMAN_RIDER_OTHER_RIDER,
        MapillaryVistasClass.OBJECT_BENCH, MapillaryVistasClass.OBJECT_FIRE_HYDRANT, MapillaryVistasClass.OBJECT_JUNCTION_BOX, 
        MapillaryVistasClass.OBJECT_MAILBOX, MapillaryVistasClass.OBJECT_PARKING_METER, MapillaryVistasClass.OBJECT_PHONE_BOOTH,
        MapillaryVistasClass.OBJECT_SIGN_ADVERTISEMENT, MapillaryVistasClass.OBJECT_SIGN_AMBIGUOUS, MapillaryVistasClass.OBJECT_SIGN_BACK, 
        MapillaryVistasClass.OBJECT_SIGN_INFORMATION, MapillaryVistasClass.OBJECT_SIGN_OTHER, MapillaryVistasClass.OBJECT_SIGN_STORE
    ], AccessibilityClass.DYNAMIC)
    .build()
)

MAPILLARY_VISTAS_ACCESSIBILITY_13: MappingSpec[MapillaryVistasClass, AccessibilityClass] = (
    MappingBuilder(
        name="mapillary_vistas_accessibility_12",
        source_enum=MapillaryVistasClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_13_CLASSES,
    )
    .map(MapillaryVistasClass.NATURE_SKY, AccessibilityClass.BACKGROUND)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_ROAD, MapillaryVistasClass.CONSTRUCTION_FLAT_ROAD_SHOULDER,
        ## markings
        MapillaryVistasClass.MARKING_CONTINUOUS_DASHED, MapillaryVistasClass.MARKING_CONTINUOUS_SOLID, MapillaryVistasClass.MARKING_CONTINUOUS_ZIGZAG,
        MapillaryVistasClass.MARKING_DISCRETE_AMBIGUOUS, MapillaryVistasClass.MARKING_DISCRETE_ARROW_LEFT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_OTHER, MapillaryVistasClass.MARKING_DISCRETE_ARROW_RIGHT,
        MapillaryVistasClass.MARKING_DISCRETE_ARROW_SPLIT_LEFT_OR_STRAIGHT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_SPLIT_RIGHT_OR_STRAIGHT, MapillaryVistasClass.MARKING_DISCRETE_ARROW_STRAIGHT,
        MapillaryVistasClass.MARKING_DISCRETE_GIVE_WAY_ROW, MapillaryVistasClass.MARKING_DISCRETE_GIVE_WAY_SINGLE, MapillaryVistasClass.MARKING_DISCRETE_HATCHED_CHEVRON, 
        MapillaryVistasClass.MARKING_DISCRETE_HATCHED_DIAGONAL, MapillaryVistasClass.MARKING_DISCRETE_OTHER_MARKING, MapillaryVistasClass.MARKING_DISCRETE_STOP_LINE, MapillaryVistasClass.MARKING_DISCRETE_SYMBOL_BICYCLE, 
        MapillaryVistasClass.MARKING_DISCRETE_SYMBOL_OTHER, MapillaryVistasClass.MARKING_DISCRETE_TEXT, 
        MapillaryVistasClass.MARKING_ONLY_CONTINUOUS_DASHED, MapillaryVistasClass.MARKING_ONLY_DISCRETE_OTHER_MARKING, MapillaryVistasClass.MARKING_ONLY_DISCRETE_TEXT,
    ], AccessibilityClass.ROAD)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_PEDESTRIAN_AREA, MapillaryVistasClass.CONSTRUCTION_FLAT_SIDEWALK,
    ], AccessibilityClass.SIDEWALK)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_STRUCTURE_BUILDING, MapillaryVistasClass.CONSTRUCTION_STRUCTURE_GARAGE
    ], AccessibilityClass.BUILDING)
    .map_many([
        MapillaryVistasClass.OBJECT_STREET_LIGHT, ### street light
        MapillaryVistasClass.OBJECT_SUPPORT_POLE, MapillaryVistasClass.OBJECT_SUPPORT_POLE_GROUP,
        MapillaryVistasClass.OBJECT_SUPPORT_TRAFFIC_SIGN_FRAME, MapillaryVistasClass.OBJECT_SUPPORT_UTILITY_POLE
    ], AccessibilityClass.POLE)
    .map_many([
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_SINGLE, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_PEDESTRIANS,
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_UPRIGHT, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_GENERAL_HORIZONTAL,
        MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_CYCLISTS, MapillaryVistasClass.OBJECT_TRAFFIC_LIGHT_OTHER
    ], AccessibilityClass.TRAFFIC_LIGHT)
    .map_many([
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_AMBIGUOUS, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_BACK,
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_DIRECTION_BACK, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_DIRECTION_FRONT,
        MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_FRONT, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_INFORMATION_PARKING, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_TEMPORARY_BACK, MapillaryVistasClass.OBJECT_TRAFFIC_SIGN_TEMPORARY_FRONT
    ], AccessibilityClass.TRAFFIC_SIGN)
    .map(MapillaryVistasClass.CONSTRUCTION_BARRIER_CURB, AccessibilityClass.CURB)
    .map(MapillaryVistasClass.CONSTRUCTION_FLAT_CURB_CUT, AccessibilityClass.CURB_RAMP)
    .map_many([
        MapillaryVistasClass.NATURE_VEGETATION
    ], AccessibilityClass.VEGETATION)
    .map_many([
        MapillaryVistasClass.NATURE_SAND, MapillaryVistasClass.NATURE_SNOW, MapillaryVistasClass.NATURE_TERRAIN
    ], AccessibilityClass.TERRAIN)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_BARRIER_AMBIGUOUS, MapillaryVistasClass.CONSTRUCTION_BARRIER_CONCRETE_BLOCK, MapillaryVistasClass.CONSTRUCTION_BARRIER_FENCE,
        MapillaryVistasClass.CONSTRUCTION_BARRIER_GUARD_RAIL, MapillaryVistasClass.CONSTRUCTION_BARRIER_OTHER_BARRIER, MapillaryVistasClass.CONSTRUCTION_BARRIER_ROAD_MEDIAN, 
        MapillaryVistasClass.CONSTRUCTION_BARRIER_ROAD_SIDE, MapillaryVistasClass.CONSTRUCTION_BARRIER_SEPARATOR, MapillaryVistasClass.CONSTRUCTION_BARRIER_WALL,
        MapillaryVistasClass.OBJECT_BIKE_RACK
    ], AccessibilityClass.STATIC)
    .map_many([
        MapillaryVistasClass.OBJECT_TRASH_CAN, MapillaryVistasClass.OBJECT_VEHICLE_BICYCLE, MapillaryVistasClass.OBJECT_VEHICLE_BOAT, MapillaryVistasClass.OBJECT_VEHICLE_BUS,
        MapillaryVistasClass.OBJECT_VEHICLE_CAR, MapillaryVistasClass.OBJECT_VEHICLE_MOTORCYCLE, MapillaryVistasClass.OBJECT_VEHICLE_TRUCK, MapillaryVistasClass.OBJECT_VEHICLE_CARAVAN, 
        MapillaryVistasClass.OBJECT_VEHICLE_ON_RAILS, MapillaryVistasClass.OBJECT_VEHICLE_OTHER_VEHICLE, MapillaryVistasClass.OBJECT_VEHICLE_TRAILER, MapillaryVistasClass.OBJECT_VEHICLE_VEHICLE_GROUP,
        MapillaryVistasClass.OBJECT_VEHICLE_WHEELED_SLOW, 
        MapillaryVistasClass.HUMAN_PERSON_INDIVIDUAL, MapillaryVistasClass.HUMAN_PERSON_PERSON_GROUP, MapillaryVistasClass.HUMAN_RIDER_BICYCLIST, MapillaryVistasClass.HUMAN_RIDER_MOTORCYCLIST, 
        MapillaryVistasClass.HUMAN_RIDER_OTHER_RIDER,
        MapillaryVistasClass.OBJECT_BENCH, MapillaryVistasClass.OBJECT_FIRE_HYDRANT, MapillaryVistasClass.OBJECT_JUNCTION_BOX, 
        MapillaryVistasClass.OBJECT_MAILBOX, MapillaryVistasClass.OBJECT_PARKING_METER, MapillaryVistasClass.OBJECT_PHONE_BOOTH,
        MapillaryVistasClass.OBJECT_SIGN_ADVERTISEMENT, MapillaryVistasClass.OBJECT_SIGN_AMBIGUOUS, MapillaryVistasClass.OBJECT_SIGN_BACK, 
        MapillaryVistasClass.OBJECT_SIGN_INFORMATION, MapillaryVistasClass.OBJECT_SIGN_OTHER, MapillaryVistasClass.OBJECT_SIGN_STORE
    ], AccessibilityClass.DYNAMIC)
    .build()
)

MAPILLARY_VISTAS_ACCESSIBILITY_3_CURB_RAMP: MappingSpec[MapillaryVistasClass, AccessibilityClass] = (
    MappingBuilder(
        name="mapillary_vistas_accessibility_3_curb_ramp",
        source_enum=MapillaryVistasClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_3_CURB_RAMP_CLASSES,
    )
    .map(MapillaryVistasClass.NATURE_SKY, AccessibilityClass.BACKGROUND)
    .map_many([
        MapillaryVistasClass.CONSTRUCTION_FLAT_PEDESTRIAN_AREA, MapillaryVistasClass.CONSTRUCTION_FLAT_SIDEWALK,
    ], AccessibilityClass.SIDEWALK)
    .map(MapillaryVistasClass.CONSTRUCTION_FLAT_CURB_CUT, AccessibilityClass.CURB_RAMP)
    .build()
)

if __name__ == "__main__":
    # Print the mapping specifications for verification
    print("Mapillary Vistas Accessibility 11 Mapping:")
    label_map_11 = MAPILLARY_VISTAS_ACCESSIBILITY_11.compile_label_map()
    print(label_map_11.shape)
    
    print("\nMapillary Vistas Accessibility 12 Mapping:")
    label_map_12 = MAPILLARY_VISTAS_ACCESSIBILITY_12.compile_label_map()
    print(label_map_12.shape)
    
    print("\nMapillary Vistas Accessibility 13 Mapping:")
    label_map_13 = MAPILLARY_VISTAS_ACCESSIBILITY_13.compile_label_map()
    print(label_map_13.shape)