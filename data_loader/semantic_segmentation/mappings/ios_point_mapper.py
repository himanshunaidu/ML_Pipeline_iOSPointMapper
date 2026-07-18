from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass
from data_loader.semantic_segmentation.taxonomies.ios_point_mapper import iOSPointMapperClass
from data_loader.semantic_segmentation.taxonomies.accessibility_class import AccessibilityClass, ACCESSIBILITY_11_CLASSES, ACCESSIBILITY_12_CLASSES, ACCESSIBILITY_13_CLASSES, ACCESSIBILITY_3_CURB_RAMP_CLASSES
from data_loader.semantic_segmentation.mappings.builder import MappingSpec, MappingBuilder

IOS_POINT_MAPPER_ACCESSIBILITY_11: MappingSpec[iOSPointMapperClass, AccessibilityClass] = (
    MappingBuilder(
        name="ios_point_mapper_accessibility_11",
        source_enum=iOSPointMapperClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_11_CLASSES,
    )
    .map(iOSPointMapperClass.BACKGROUND, AccessibilityClass.BACKGROUND)
    .map(iOSPointMapperClass.ROAD, AccessibilityClass.ROAD)
    .map(iOSPointMapperClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map(iOSPointMapperClass.BUILDING, AccessibilityClass.BUILDING)
    .map(iOSPointMapperClass.POLE, AccessibilityClass.POLE)
    .map(iOSPointMapperClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map(iOSPointMapperClass.TRAFFIC_SIGN, AccessibilityClass.TRAFFIC_SIGN)
    .map(iOSPointMapperClass.VEGETATION, AccessibilityClass.VEGETATION)
    .map(iOSPointMapperClass.TERRAIN, AccessibilityClass.TERRAIN)
    .map(iOSPointMapperClass.STATIC, AccessibilityClass.STATIC)
    .map(iOSPointMapperClass.DYNAMIC, AccessibilityClass.DYNAMIC)
    .build()
)

IOS_POINT_MAPPER_ACCESSIBILITY_12: MappingSpec[iOSPointMapperClass, AccessibilityClass] = (
    MappingBuilder(
        name="ios_point_mapper_accessibility_12",
        source_enum=iOSPointMapperClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_12_CLASSES,
    )
    .map_many([
        iOSPointMapperClass.BACKGROUND,
        iOSPointMapperClass.CURB
    ], AccessibilityClass.BACKGROUND)
    .map(iOSPointMapperClass.ROAD, AccessibilityClass.ROAD)
    .map(iOSPointMapperClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map(iOSPointMapperClass.BUILDING, AccessibilityClass.BUILDING)
    .map(iOSPointMapperClass.POLE, AccessibilityClass.POLE)
    .map(iOSPointMapperClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map(iOSPointMapperClass.TRAFFIC_SIGN, AccessibilityClass.TRAFFIC_SIGN)
    .map_many([
        iOSPointMapperClass.CURB_RAMP, iOSPointMapperClass.TACTILE_PAVING # Mostly tactile pavings in the dataset are on curb ramps
    ], AccessibilityClass.CURB_RAMP)
    .map(iOSPointMapperClass.VEGETATION, AccessibilityClass.VEGETATION)
    .map(iOSPointMapperClass.TERRAIN, AccessibilityClass.TERRAIN)
    .map(iOSPointMapperClass.STATIC, AccessibilityClass.STATIC)
    .map(iOSPointMapperClass.DYNAMIC, AccessibilityClass.DYNAMIC)
    .build()
)

IOS_POINT_MAPPER_ACCESSIBILITY_13: MappingSpec[iOSPointMapperClass, AccessibilityClass] = (
    MappingBuilder(
        name="ios_point_mapper_accessibility_13",
        source_enum=iOSPointMapperClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_13_CLASSES,
    )
    .map(iOSPointMapperClass.BACKGROUND, AccessibilityClass.BACKGROUND)
    .map(iOSPointMapperClass.ROAD, AccessibilityClass.ROAD)
    .map(iOSPointMapperClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map(iOSPointMapperClass.BUILDING, AccessibilityClass.BUILDING)
    .map(iOSPointMapperClass.POLE, AccessibilityClass.POLE)
    .map(iOSPointMapperClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map(iOSPointMapperClass.TRAFFIC_SIGN, AccessibilityClass.TRAFFIC_SIGN)
    .map(iOSPointMapperClass.CURB, AccessibilityClass.CURB)
    .map_many([
        iOSPointMapperClass.CURB_RAMP, iOSPointMapperClass.TACTILE_PAVING # Mostly tactile pavings in the dataset are on curb ramps
    ], AccessibilityClass.CURB_RAMP)
    .map(iOSPointMapperClass.VEGETATION, AccessibilityClass.VEGETATION)
    .map(iOSPointMapperClass.TERRAIN, AccessibilityClass.TERRAIN)
    .map(iOSPointMapperClass.STATIC, AccessibilityClass.STATIC)
    .map(iOSPointMapperClass.DYNAMIC, AccessibilityClass.DYNAMIC)
    .build()
)

IOS_POINT_MAPPER_ACCESSIBILITY_3_CURB_RAMP: MappingSpec[iOSPointMapperClass, AccessibilityClass] = (
    MappingBuilder(
        name="ios_point_mapper_accessibility_3_curb_ramp",
        source_enum=iOSPointMapperClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_3_CURB_RAMP_CLASSES,
    )
    .map(iOSPointMapperClass.BACKGROUND, AccessibilityClass.BACKGROUND)
    .map(iOSPointMapperClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map_many([
        iOSPointMapperClass.CURB_RAMP, iOSPointMapperClass.TACTILE_PAVING # Mostly tactile pavings in the dataset are on curb ramps
    ], AccessibilityClass.CURB_RAMP)
    .build()
)

if __name__ == "__main__":
    # Print the mapping specifications for verification
    print("iOS Point Mapper to Accessibility Class Mapping (11 classes):")
    label_map_11 = IOS_POINT_MAPPER_ACCESSIBILITY_11.compile_label_map()
    print(label_map_11.shape)
    
    print("\niOS Point Mapper to Accessibility Class Mapping (12 classes):")
    label_map_12 = IOS_POINT_MAPPER_ACCESSIBILITY_12.compile_label_map()
    print(label_map_12.shape)
    
    print("\niOS Point Mapper to Accessibility Class Mapping (13 classes):")
    label_map_13 = IOS_POINT_MAPPER_ACCESSIBILITY_13.compile_label_map()
    print(label_map_13.shape)