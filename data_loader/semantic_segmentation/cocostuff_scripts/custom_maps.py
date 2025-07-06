"""
This script stores all the custom dictionaries for COCO-Stuff used for mapping classes in semantic segmentation tasks.
"""

# This dictionary stores the mapping of relevant COCO-Stuff dataset classes to their respective names.
cocoStuff_dict = {0:'background', 1:'person', 2:'bicycle', 3:'car', 4:'motorcycle', 6:'bus', 7:'train', 8:'truck',
                  10:'traffic light', 11:'fire hydrant', 12:'street sign', 13:'stop sign', 14:'parking meter',
                  15:'bench', # 33: 'suitcase', 41:'skateboard', 
                  64:'potted plant', 92:'banner', 94:'branch',
                  96:'building-other', 97:'bush', 99:'cage', 100:'cardboard', 111:'dirt', 113:'fence', 
                #   115:'floor-other', 116:'floor-stone', 
                  124:'grass', 125:'gravel', 126:'ground-other', 
                  128:'house', 129:'leaves', # 130:'light', 
                  132: 'metal', 134:'moss', 136:'mud', 140:'pavement', 142:'plant-other', 144:'platform',
                  145:'playfield', 146:'railing', 147:'railroad', 149:'road', 150:'rock', 151:'roof', 154:'sand', 159:'snow',
                  161:'stairs', 162:'stone', 164:'structural-other', 169:'tree', 171: 'wall-brick', 172:'wall-concrete', 
                  173:'wall-other', 174:'wall-panel', 175:'wall-stone', 176:'wall-tile', 177:'wall-wood', # 178:'water-other', 
                  182:'wood' }

# This dictionary maps edge_mapping and cityscapes classes to COCO-Stuff classes.
## Not in use, but kept for reference.
cos2cocoStuff_dict = {0:149, 1:140, 2:96, 3:173, 4:113, 5:0, 6:10, 7:13, 8:129, 9:124,
                      10:0, 11:1, 12:1, 13:3, 14:8, 15:6, 16:7, 17:2, 18:2, 19:0}

# The following dictionary is to map the relevant cocostuff classes to a continuous set of labels.
## This specific dictionary takes all the 53 relevant classes from COCO-Stuff and maps them one-by-one to a continuous set of labels.
cocoStuff_continuous_53_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7,
                  10:8, 11:9, 12:10, 13:11, 14:12,
                  15:13, 33:14, 41:15, 64:16, 92:17, 94:18,
                  96:19, 97:20, 99:21, 100:22, 111:23, 113:24, 
                  115:25, 116:26, 124:27, 125:28, 126:29, 128:30,
                  129:31, 130:32, 134:33, 136:34, 140:35, 142:36, 144:37,
                  145:38, 146:39, 147:40, 149:41, 150:42, 151:43, 154:44, 159:45,
                  161:46, 162:47, 164:48, 169:49, 171: 50, 172:50, 
                  173:50, 174:50, 175:50, 176:50, 177:50, 178:51, 182:52 }

# The following dictionary is to map the relevant cocostuff classes to a continuous set of labels.
## This specific dictionary takes all the 53 relevant classes and maps them to 35 continuous labels.
## Thus, in some cases, multiple classes are mapped to the same label.
## Multiple classes to one include: traffic sign, vegetation, terrain
### traffic sign (10): 12, 13
### vegetation (15): 94, 97, 129, 142, 169
### terrain (19): 111, 124, 125, 126 (cancelled), 134, 136, 154, 159
### building (16): 96, 128
### wall (33): 171, 172, 173, 174, 175, 176, 177
cocoStuff_continuous_35_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7,
                  10:8, 11:9, 
                  12:10, 13:10, # traffic sign
                  14:11, 15:12, 64:13, 92:14, 
                  94:15, 97:15, 129:15, 142:15, 169:15, # vegetation
                  96:16, 128:16, # building
                  99:17, 100:18, 
                  111:19, 124:19, 125:19, 126:0, 134:19, 136:19, 154:19, 159:19, # terrain
                  113:20, 132: 21, 140:22, 144:23,
                  145:24, 146:25, 147:26, 149:27, 150:28, 151:29,
                  161:30, 162:31, 164:32, 
                  171:33, 172:33, 173:33, 174:33, 175:33, 176:33, 177:33, # wall
                  182:34 }