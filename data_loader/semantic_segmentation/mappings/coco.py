from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass
from data_loader.semantic_segmentation.taxonomies.coco import CocoStuffClass
from data_loader.semantic_segmentation.taxonomies.accessibility_class import AccessibilityClass, ACCESSIBILITY_11_CLASSES, ACCESSIBILITY_12_CLASSES, ACCESSIBILITY_13_CLASSES
from data_loader.semantic_segmentation.mappings.builder import MappingSpec, MappingBuilder

COCO_STUFF_ACCESSIBILITY_11: MappingSpec[CocoStuffClass, AccessibilityClass] = (
    MappingBuilder(
        name="coco_stuff_accessibility_11",
        source_enum=CocoStuffClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_11_CLASSES,
    )
    .map(CocoStuffClass.BACKGROUND, AccessibilityClass.BACKGROUND)
    .map(CocoStuffClass.ROAD, AccessibilityClass.ROAD)
    .map(CocoStuffClass.PAVEMENT, AccessibilityClass.SIDEWALK)
    .map_many([CocoStuffClass.BUILDING_OTHER, CocoStuffClass.HOUSE], AccessibilityClass.BUILDING)
    .map(CocoStuffClass.METAL, AccessibilityClass.POLE)
    .map(CocoStuffClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map_many([CocoStuffClass.STREET_SIGN, CocoStuffClass.STOP_SIGN], AccessibilityClass.TRAFFIC_SIGN)
    .map_many([CocoStuffClass.BRANCH, CocoStuffClass.BUSH, CocoStuffClass.LEAVES, CocoStuffClass.PLANT_OTHER, CocoStuffClass.TREE], AccessibilityClass.VEGETATION)
    .map_many([CocoStuffClass.DIRT, CocoStuffClass.GRASS, CocoStuffClass.GRAVEL, CocoStuffClass.GROUND_OTHER, CocoStuffClass.MOSS, CocoStuffClass.MUD, CocoStuffClass.SAND, CocoStuffClass.SNOW], AccessibilityClass.TERRAIN)
    .map_many([CocoStuffClass.WALL_OTHER, CocoStuffClass.WALL_BRICK, CocoStuffClass.WALL_CONCRETE, CocoStuffClass.WALL_PANEL, CocoStuffClass.WALL_STONE, CocoStuffClass.WALL_TILE, CocoStuffClass.WALL_WOOD, CocoStuffClass.WATER_OTHER, CocoStuffClass.WOOD, CocoStuffClass.SKY, CocoStuffClass.CLOUDS, CocoStuffClass.PLAYINGFIELD, CocoStuffClass.FENCE], AccessibilityClass.STATIC)
    .map_many([CocoStuffClass.PERSON, CocoStuffClass.BICYCLE, CocoStuffClass.CAR, CocoStuffClass.MOTORCYCLE, CocoStuffClass.BUS, CocoStuffClass.TRAIN, CocoStuffClass.TRUCK], AccessibilityClass.DYNAMIC)
    .build()
)

COCO_STUFF_ACCESSIBILITY_12: MappingSpec[CocoStuffClass, AccessibilityClass] = (
    MappingBuilder(
        name="coco_stuff_accessibility_12",
        source_enum=CocoStuffClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_12_CLASSES,
    )
    .map(CocoStuffClass.BACKGROUND, AccessibilityClass.BACKGROUND)
    .map(CocoStuffClass.ROAD, AccessibilityClass.ROAD)
    .map(CocoStuffClass.PAVEMENT, AccessibilityClass.SIDEWALK)
    .map_many([CocoStuffClass.BUILDING_OTHER, CocoStuffClass.HOUSE], AccessibilityClass.BUILDING)
    .map(CocoStuffClass.METAL, AccessibilityClass.POLE)
    .map(CocoStuffClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map_many([CocoStuffClass.STREET_SIGN, CocoStuffClass.STOP_SIGN], AccessibilityClass.TRAFFIC_SIGN)
    .mark_unrepresented(AccessibilityClass.CURB_RAMP, reason="No curb ramp classes in COCO-Stuff")
    .map_many([CocoStuffClass.BRANCH, CocoStuffClass.BUSH, CocoStuffClass.LEAVES, CocoStuffClass.PLANT_OTHER, CocoStuffClass.TREE], AccessibilityClass.VEGETATION)
    .map_many([CocoStuffClass.DIRT, CocoStuffClass.GRASS, CocoStuffClass.GRAVEL, CocoStuffClass.GROUND_OTHER, CocoStuffClass.MOSS, CocoStuffClass.MUD, CocoStuffClass.SAND, CocoStuffClass.SNOW], AccessibilityClass.TERRAIN)
    .map_many([CocoStuffClass.WALL_OTHER, CocoStuffClass.WALL_BRICK, CocoStuffClass.WALL_CONCRETE, CocoStuffClass.WALL_PANEL, CocoStuffClass.WALL_STONE, CocoStuffClass.WALL_TILE, CocoStuffClass.WALL_WOOD, CocoStuffClass.WATER_OTHER, CocoStuffClass.WOOD, CocoStuffClass.SKY, CocoStuffClass.CLOUDS, CocoStuffClass.PLAYINGFIELD, CocoStuffClass.FENCE], AccessibilityClass.STATIC)
    .map_many([CocoStuffClass.PERSON, CocoStuffClass.BICYCLE, CocoStuffClass.CAR, CocoStuffClass.MOTORCYCLE, CocoStuffClass.BUS, CocoStuffClass.TRAIN, CocoStuffClass.TRUCK], AccessibilityClass.DYNAMIC)
    .build()
)

COCO_STUFF_ACCESSIBILITY_13: MappingSpec[CocoStuffClass, AccessibilityClass] = (
    MappingBuilder(
        name="coco_stuff_accessibility_13",
        source_enum=CocoStuffClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_13_CLASSES,
    )
    .map(CocoStuffClass.BACKGROUND, AccessibilityClass.BACKGROUND)
    .map(CocoStuffClass.ROAD, AccessibilityClass.ROAD)
    .map(CocoStuffClass.PAVEMENT, AccessibilityClass.SIDEWALK)
    .map_many([CocoStuffClass.BUILDING_OTHER, CocoStuffClass.HOUSE], AccessibilityClass.BUILDING)
    .map(CocoStuffClass.METAL, AccessibilityClass.POLE)
    .map(CocoStuffClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map_many([CocoStuffClass.STREET_SIGN, CocoStuffClass.STOP_SIGN], AccessibilityClass.TRAFFIC_SIGN)
    .mark_unrepresented(AccessibilityClass.CURB, reason="No curb classes in COCO-Stuff")
    .mark_unrepresented(AccessibilityClass.CURB_RAMP, reason="No curb ramp classes in COCO-Stuff")
    .map_many([CocoStuffClass.BRANCH, CocoStuffClass.BUSH, CocoStuffClass.LEAVES, CocoStuffClass.PLANT_OTHER, CocoStuffClass.TREE], AccessibilityClass.VEGETATION)
    .map_many([CocoStuffClass.DIRT, CocoStuffClass.GRASS, CocoStuffClass.GRAVEL, CocoStuffClass.GROUND_OTHER, CocoStuffClass.MOSS, CocoStuffClass.MUD, CocoStuffClass.SAND, CocoStuffClass.SNOW], AccessibilityClass.TERRAIN)
    .map_many([CocoStuffClass.WALL_OTHER, CocoStuffClass.WALL_BRICK, CocoStuffClass.WALL_CONCRETE, CocoStuffClass.WALL_PANEL, CocoStuffClass.WALL_STONE, CocoStuffClass.WALL_TILE, CocoStuffClass.WALL_WOOD, CocoStuffClass.WATER_OTHER, CocoStuffClass.WOOD, CocoStuffClass.SKY, CocoStuffClass.CLOUDS, CocoStuffClass.PLAYINGFIELD, CocoStuffClass.FENCE], AccessibilityClass.STATIC)
    .map_many([CocoStuffClass.PERSON, CocoStuffClass.BICYCLE, CocoStuffClass.CAR, CocoStuffClass.MOTORCYCLE, CocoStuffClass.BUS, CocoStuffClass.TRAIN, CocoStuffClass.TRUCK], AccessibilityClass.DYNAMIC)
    .build()
)

if __name__ == "__main__":
    # Print the mapping specifications for verification
    print("CocoStuff to Accessibility Class Mapping (11 classes):")
    label_map_11 = COCO_STUFF_ACCESSIBILITY_11.compile_label_map()
    print(label_map_11.shape)
    
    print("\nCocoStuff to Accessibility Class Mapping (12 classes):")
    label_map_12 = COCO_STUFF_ACCESSIBILITY_12.compile_label_map()
    print(label_map_12.shape)
    
    print("\nCocoStuff to Accessibility Class Mapping (13 classes):")
    label_map_13 = COCO_STUFF_ACCESSIBILITY_13.compile_label_map()
    print(label_map_13.shape)