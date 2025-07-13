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
edge_mapping_to_cocoStuff_custom_35_dict = {0:27, 1:22, 2:16, 3:33, 4:20, 5:21, 6:8, 7:10, 8:15, 9:19,
                            10:0, 11:1, 12:1, 13:3, 14:7, 15:5, 16:6, 17:4, 18:2, 19:0}