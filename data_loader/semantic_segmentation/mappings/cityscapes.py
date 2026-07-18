from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass
from data_loader.semantic_segmentation.taxonomies.cityscapes import CityscapesClass
from data_loader.semantic_segmentation.taxonomies.accessibility_class import AccessibilityClass, ACCESSIBILITY_11_CLASSES, ACCESSIBILITY_12_CLASSES, ACCESSIBILITY_13_CLASSES
from data_loader.semantic_segmentation.mappings.builder import MappingSpec, MappingBuilder

CITYSCAPES_ACCESSIBILITY_11: MappingSpec[CityscapesClass, AccessibilityClass] = (
    MappingBuilder(
        name="cityscapes_accessibility_11",
        source_enum=CityscapesClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_11_CLASSES,
    )
    .map_many([CityscapesClass.SKY, CityscapesClass.BACKGROUND], AccessibilityClass.BACKGROUND)
    .map(CityscapesClass.ROAD, AccessibilityClass.ROAD)
    .map(CityscapesClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map(CityscapesClass.BUILDING, AccessibilityClass.BUILDING)
    .map(CityscapesClass.POLE, AccessibilityClass.POLE)
    .map(CityscapesClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map(CityscapesClass.TRAFFIC_SIGN, AccessibilityClass.TRAFFIC_SIGN)
    .map(CityscapesClass.VEGETATION, AccessibilityClass.VEGETATION)
    .map(CityscapesClass.TERRAIN, AccessibilityClass.TERRAIN)
    .map_many([CityscapesClass.WALL, CityscapesClass.FENCE], AccessibilityClass.STATIC)
    .map_many([CityscapesClass.PERSON, CityscapesClass.RIDER, CityscapesClass.CAR, CityscapesClass.TRUCK, CityscapesClass.BUS, CityscapesClass.TRAIN, CityscapesClass.MOTORCYCLE, CityscapesClass.BICYCLE], AccessibilityClass.DYNAMIC)
    .build()
)

CITYSCAPES_ACCESSIBILITY_12: MappingSpec[CityscapesClass, AccessibilityClass] = (
    MappingBuilder(
        name="cityscapes_accessibility_12",
        source_enum=CityscapesClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_12_CLASSES,
    )
    .map_many([CityscapesClass.SKY, CityscapesClass.BACKGROUND], AccessibilityClass.BACKGROUND)
    .map(CityscapesClass.ROAD, AccessibilityClass.ROAD)
    .map(CityscapesClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map(CityscapesClass.BUILDING, AccessibilityClass.BUILDING)
    .map(CityscapesClass.POLE, AccessibilityClass.POLE)
    .map(CityscapesClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map(CityscapesClass.TRAFFIC_SIGN, AccessibilityClass.TRAFFIC_SIGN)
    .mark_unrepresented(AccessibilityClass.CURB_RAMP, reason="No curb ramp classes in Cityscapes")
    .map(CityscapesClass.VEGETATION, AccessibilityClass.VEGETATION)
    .map(CityscapesClass.TERRAIN, AccessibilityClass.TERRAIN)
    .map_many([CityscapesClass.WALL, CityscapesClass.FENCE], AccessibilityClass.STATIC)
    .map_many([CityscapesClass.PERSON, CityscapesClass.RIDER, CityscapesClass.CAR, CityscapesClass.TRUCK, CityscapesClass.BUS, CityscapesClass.TRAIN, CityscapesClass.MOTORCYCLE, CityscapesClass.BICYCLE], AccessibilityClass.DYNAMIC)
    .build()
)

CITYSCAPES_ACCESSIBILITY_13: MappingSpec[CityscapesClass, AccessibilityClass] = (
    MappingBuilder(
        name="cityscapes_accessibility_13",
        source_enum=CityscapesClass,
        target_enum=AccessibilityClass,
        target_classes=ACCESSIBILITY_13_CLASSES,
    )
    .map_many([CityscapesClass.SKY, CityscapesClass.BACKGROUND], AccessibilityClass.BACKGROUND)
    .map(CityscapesClass.ROAD, AccessibilityClass.ROAD)
    .map(CityscapesClass.SIDEWALK, AccessibilityClass.SIDEWALK)
    .map(CityscapesClass.BUILDING, AccessibilityClass.BUILDING)
    .map(CityscapesClass.POLE, AccessibilityClass.POLE)
    .map(CityscapesClass.TRAFFIC_LIGHT, AccessibilityClass.TRAFFIC_LIGHT)
    .map(CityscapesClass.TRAFFIC_SIGN, AccessibilityClass.TRAFFIC_SIGN)
    .mark_unrepresented(AccessibilityClass.CURB, reason="No curb classes in Cityscapes")
    .mark_unrepresented(AccessibilityClass.CURB_RAMP, reason="No curb ramp classes in Cityscapes")
    .map(CityscapesClass.VEGETATION, AccessibilityClass.VEGETATION)
    .map(CityscapesClass.TERRAIN, AccessibilityClass.TERRAIN)
    .map_many([CityscapesClass.WALL, CityscapesClass.FENCE], AccessibilityClass.STATIC)
    .map_many([CityscapesClass.PERSON, CityscapesClass.RIDER, CityscapesClass.CAR, CityscapesClass.TRUCK, CityscapesClass.BUS, CityscapesClass.TRAIN, CityscapesClass.MOTORCYCLE, CityscapesClass.BICYCLE], AccessibilityClass.DYNAMIC)
    .build()
)

if __name__ == "__main__":
    # Print the mapping specifications for verification
    print("Cityscapes to Accessibility Class Mapping (11 classes):")
    label_map_11 = CITYSCAPES_ACCESSIBILITY_11.compile_label_map()
    print(label_map_11.shape)
    
    print("\nCityscapes to Accessibility Class Mapping (12 classes):")
    label_map_12 = CITYSCAPES_ACCESSIBILITY_12.compile_label_map()
    print(label_map_12.shape)
    
    print("\nCityscapes to Accessibility Class Mapping (13 classes):")
    label_map_13 = CITYSCAPES_ACCESSIBILITY_13.compile_label_map()
    print(label_map_13.shape)