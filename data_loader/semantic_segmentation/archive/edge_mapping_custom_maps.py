"""
This script stores all the custom dictionaries for Custom Edge Mapping used for mapping classes in semantic segmentation tasks.
"""
# Mapping from edge mapping classes to **custom** cocostuff classes (53 classes)
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
edge_mapping_to_cocoStuff_custom_53_dict = {0:41, 1:35, 2:19, 3:50, 4:24, 5:0, 6:8, 7:11, 8:31, 9:27,
                            10:0, 11:1, 12:1, 13:3, 14:12, 15:5, 16:6, 17:2, 18:2, 19:0}

# Mapping from edge mapping classes to **custom** cocostuff classes (35 classes)
## This customization of cocostuff classes comes from edge mapping repository
## done to map the fewer relevant classes to a continuous range of classes
edge_mapping_to_cocoStuff_custom_35_dict = {0:26, 1:21, 2:15, 3:32, 4:19, 5:20, 6:7, 7:9, 8:14, 9:18,
    10:255, 11:0, 12:0, 13:2, 14:6, 15:4, 16:5, 17:3, 18:1, 19:255}