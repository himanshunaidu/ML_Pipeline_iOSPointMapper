"""
This script stores all the custom dictionaries for Custom iOSPointMapper used for mapping classes in semantic segmentation tasks.
"""
ios_point_mapper_dict = {
    0: 'background', 1: 'bicycle', 2: 'bike rack', 3: 'bridge', 4: 'building',
    5: 'bus', 6: 'car', 7: 'dynamic', 8: 'fence', 9: 'ground',
    10: 'guard rail', 11: 'motorcycle', 12: 'parking', 13: 'person',
    14: 'pole', 15: 'rail track', 16: 'rider', 17: 'road',
    18: 'sidewalk', 19: 'sky', 20: 'static',
    21: 'terrain', 22: 'traffic light', 23: 'traffic sign',
    24: 'train', 25: 'truck', 26: 'tunnel',
    27: 'vegetation', 28: 'wall'
}

# Mapping from ios_point_mapper classes to **custom** cocostuff classes
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
ios_point_mapper_to_cocoStuff_custom_53_dict = {0:0, 1:2, 2:0, 3:0, 4:19, 5:5, 6:3, 7:0, 8:24, 9:0, 10:0, 
                                    11:2, 12:0, 13:1, 14:0, 15:0, 16:1, 17:41, 18:35, 19:0, 20:0, 
                                    21:27, 22:8, 23:11, 24:6, 25:12, 26:0, 27:31, 28:50}

# Mapping from ios_point_mapper classes to **custom** cocostuff classes
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
ios_point_mapper_to_cocoStuff_custom_35_dict = {0:255, 1:1, 2:255, 3:255, 4:15, 5:4, 6:2, 7:255, 8:19, 9:255,
                                    10:255, 11:3, 12:255, 13:0, 14:20, 15:255, 16:0, 17:26, 18:21,
                                    19:255, 20:255, 21:18, 22:7, 23:9, 24:5, 25:6, 26:255, 27:14, 28:32}

# Mapping from ios_point_mapper classes to cityscapes classes
ios_point_mapper_to_cityscapes_dict = {
    0:255, 1:18, 2:255, 3:255, 4:2, 5:15, 6:13, 7:255, 8:4, 9:255, 10:255, 11:17, 12:255,
    13:11, 14:5, 15:255, 16:12, 17:0, 18:1, 19:10, 20:255, 21:9, 22:6, 23:7, 24:16, 25:14, 
    26:255, 27:8, 28:3}