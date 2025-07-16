"""
This script stores all the custom dictionaries for Custom iOSPointMapper used for mapping classes in semantic segmentation tasks.
"""
ios_point_mapper_dict = {
    0: "background",
    1: "bicycle",
    2: "bike rack",
    3: "bridge",
    4: "building",
    5: "bus",
    6: "car",
    7: "crosswalk",
    8: "curb",
    9: "curb ramp",
    10: "dynamic",
    11: "fence",
    12: "ground",
    13: "guard rail",
    14: "license plate",
    15: "motorcycle",
    16: "parking",
    17: "person",
    18: "pole",
    19: "rail track",
    20: "rider",
    21: "road",
    22: "sidewalk",
    23: "sky",
    24: "static",
    25: "tactile paving",
    26: "terrain",
    27: "traffic light",
    28: "traffic sign",
    29: "train",
    30: "truck",
    31: "tunnel",
    32: "vegetation",
    33: "wall"
}


# cross-walk is treated as road
# curb, curb ramp and tactile paving is treated as sidewalk
# license plate is treated as car
ios_point_mapper_to_cocoStuff_custom_35_dict = {0:255, 1:1, 2:255, 3:255, 4:15, 5:4, 6:2, 7:26, 8:21, 9:21, 10:255,
    11:19, 12:255, 13:255, 14:2, 15:3, 16:255, 17:0, 18:20, 19:255,
    20:0, 21:26, 22:21, 23:255, 24:255, 25:21, 26:18, 27:7, 28:9, 29:5,
    30:6, 31:255, 32:14, 33:32}

ios_point_mapper_to_cocoStuff_custom_11_dict = {0:10, 1:9, 2:255, 3:255, 4:2, 5:9, 6:9, 7:0, 8:1, 9:1, 10:9, 
    11:8, 12:8, 13:8, 14:9, 15:9, 16:8, 17:9, 18:3, 19:8, 20:9,
    21:0, 22:1, 23:8, 24:8, 25:1, 26:7, 27:4, 28:5, 29:9, 30:9,
    31:255, 32:6, 33:8}

ios_point_mapper_to_cocoStuff_custom_9_dict = {0:8, 1:7, 2:255, 3:255, 4:2, 5:7, 6:7, 7:0, 8:1, 9:1, 10:7, 
    11:6, 12:6, 13:6, 14:7, 15:7, 16:6, 17:7, 18:3, 19:6, 20:7,
    21:0, 22:1, 23:6, 24:6, 25:1, 26:6, 27:4, 28:5, 29:7, 30:7,
    31:255, 32:6, 33:6}

ios_point_mapper_to_cityscapes_dict = {
    0:255, 1:18, 2:255, 3:255, 4:2, 5:15, 6:13, 7:0, 8:1, 9:1, 10:255, 11:4, 12:255, 13:255, 14:13, 15:17,
    16:255, 17:11, 18:5, 19:255, 20:12, 21:0, 22:1, 23:10, 24:255, 25:1, 26:9, 27:6, 28:7, 29:16, 30:14, 
    31:255, 32:8, 33:3
}