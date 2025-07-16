"""
This script stores all the custom dictionaries for COCO-Stuff used for mapping classes in semantic segmentation tasks.
"""

"""
This script stores all the custom dictionaries for COCO-Stuff used for mapping classes in semantic segmentation tasks.
"""

# This dictionary stores the mapping of relevant COCO-Stuff dataset classes to their respective names.
cocoStuff_dict = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 5:'bus', 6:'train', 7:'truck',
                        9:'traffic light', 10:'fire hydrant', 11:'street sign', 12:'stop sign', 13:'parking meter',
                        14:'bench', 32: 'suitcase', 40:'skateboard', 
                        63:'potted plant', 91:'banner', 93:'branch',
                        95:'building-other', 96:'bush', 98:'cage', 99:'cardboard', 110:'dirt', 112:'fence',
                        114:'floor-other', 115:'floor-stone',
                        123:'grass', 124:'gravel', 125:'ground-other',
                        127:'house', 128:'leaves', 129:'light',
                        131: 'metal', 133:'moss', 135:'mud', 139:'pavement', 141:'plant-other', 143:'platform',
                        144:'playingfield', 145:'railing', 146:'railroad', 148:'road', 149:'rock', 150:'roof', 153:'sand', 158:'snow',
                        160:'stairs', 161:'stone', 163:'structural-other', 168:'tree', 170: 'wall-brick', 171:'wall-concrete', 
                        172:'wall-other', 173:'wall-panel', 174:'wall-stone', 175:'wall-tile', 176:'wall-wood', 177:'water-other', 
                        181:'wood',
                        156: 'sky', 105: 'clouds'} # Extra class that are frequently present in outdoor environments.

# This dictionary maps edge_mapping and cityscapes classes to COCO-Stuff classes.
# Not in use, but kept for reference.
cos2cocoStuff_dict = {0:148, 1:139, 2:95, 3:172, 4:112, 5:131, 6:9, 7:12, 8:128, 9:123, # Having metal as 5 (pole) is not ideal, but it is the only way to keep the mapping consistent.
                            10:255, 11:0, 12:0, 13:2, 14:7, 15:5, 16:6, 17:1, 18:1, 19:255}

# The following dictionary is to map the relevant cocostuff classes to a continuous set of labels.
## This specific dictionary takes all the 53 relevant classes from COCO-Stuff and maps them one-by-one to a continuous set of labels.
cocoStuff_continuous_53_dict = {
    0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6,
    9:7, 10:8, 11:9, 12:10, 13:11,
    14:12, 32:13, 40:14,
    63:15, 91:16, 93:17, 
    95:18, 96:19, 98:20, 99:21, 110:22, 112:23,
    114:24, 115:25,
    123:26, 124:27, 125:28,
    127:29, 128:30, 129:31,
    131:32, 133:33, 135:34, 139:35, 141:36, 143:37,
    144:38, 145:39, 146:40, 148:41, 149:42, 150:43, 153:44, 158:45,
    160:46, 161:47, 163:48, 168:49, 170:50, 171:50,
    172:50, 173:50, 174:50, 175:50, 176:50, 177:50, 181:51,
    255:52  # Background is mapped to 255, which is not used in the continuous labels.
}

# The following dictionary is to map the relevant cocostuff classes to a continuous set of labels.
## This specific dictionary takes all the relevant classes and maps them to 35 continuous labels.
## Thus, in some cases, multiple classes are mapped to the same label.
## Multiple classes to one include: traffic sign, vegetation, terrain
### traffic sign (9): 11, 12
### vegetation (14): 93, 96, 128, 141, 168
### terrain (18): 110, 123, 124, 125 (cancelled), 133, 135, 153, 158
### building (15): 95, 127
### wall (32): 170, 171, 172, 173, 174, 175, 176
cocoStuff_continuous_35_dict = {
    0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6,
    9:7, 10:8,
    11:9, 12:9, # traffic sign
    13:10, 14:11, 63:12, 91:13,
    93:14, 96:14, 128:14, 141:14, 168:14, # vegetation
    95:15, 127:15, # building
    98:16, 99:17,
    110:18, 123:18, 124:18, 133:18, 135:18, 153:18, 158:18, # terrain
    112:19, 131:20, 139:21, 143:22,
    144:23, 145:24, 146:25, 148:26, 149:27, 150:28,
    160:29, 161:30, 163:31,
    170:32, 171:32, 172:32, 173:32, 174:32, 175:32, 176:32, # wall
    181:33, 255:34  # Background is mapped to 255, which is not used in the continuous labels.
}

# The following dictionary is to map a very small relevant subset of cocostuff classes to a continuous set of labels.
## Main classes: road, sidewalk, building, pole, traffic light, traffic sign, 
## Context classes: vegetation, terrain, background
## Extra static classes (high proportion): sky, clouds, playingfield, fence, wall-concrete
## Extra dynamic classes (high proportion): person, bus, train, car, truck, motorcycle
cocoStuff_continuous_11_dict = {
    148:0, 139:1, # road, sidewalk
    95:2, 127:2, # building
    131:3, 9:4, # pole, traffic light
    11:5, 12:5, # traffic sign
    93:6, 96:6, 128:6, 141:6, 168:6, # vegetation
    110:7, 123:7, 124:7, 125:7, 133:7, 135:7, 153:7, 158:7, # terrain
    172:8, 173:8, 174:8, 175:8, 176:8, 177:8, 181:8, 156:8, 105:8, 144:8, 112:8, # static classes
    0:9, 1:9, 2:9, 3:9, 5:9, 6:9, 7:9, # dynamic classes
    255:10  # Background is mapped to 255, which is not used in the continuous labels.
}
# Weight mapping for the continuous set of labels
cocoStuff_continuous_11_weights = [1.5, 3.0, 1.5, 2.5, 2.0, 2.0, 0.5, 0.5, 0.3, 0.3, 0.2] # Arbitrary weights for the classes in cocoStuff_continuous_11_dict

# The following dictionary is to map a very small relevant subset of cocostuff classes to a continuous set of labels.
## Main classes: road, sidewalk, building, pole, traffic light, traffic sign
## Extra static classes (high proportion): vegetation, terrain, sky, clouds, playingfield, fence, wall-concrete
## Extra dynamic classes (high proportion): person, bus, train, car, truck, motorcycle
cocoStuff_continuous_9_dict = {
    148:0, 139:1, # road, sidewalk
    95:2, 127:2, # building
    131:3, 9:4, # pole, traffic light
    11:5, 12:5, # traffic sign
    93:6, 96:6, 128:6, 141:6, 168:6, # vegetation (static)
    110:6, 123:6, 124:6, 125:6, 133:6, 135:6, 153:6, 158:6, # terrain (static)
    172:6, 173:6, 174:6, 175:6, 176:6, 177:6, 181:6, 156:6, 105:6, 144:6, 112:6, # static classes
    0:7, 1:7, 2:7, 3:7, 5:7, 6:7, 7:7, # dynamic classes
    255:8  # Background is mapped to 255, which is not used in the continuous labels.
}
# Weight mapping for the continuous set of labels
cocoStuff_continuous_9_weights = [1.5, 3.0, 1.5, 2.5, 2.0, 2.0, 0.1, 0.1, 0.2] # Arbitrary weights for the classes in cocoStuff_continuous_9_dict

# Map for cityscapes. Will later add remaining classes. 
cocoStuff_cityscapes_dict = {
    148: 0, 139: 1,
    95: 2, 127: 2,
    131: 5,
    9: 6,
    11: 7, 12: 7
}

# Class proportions for the COCO-Stuff dataset
cocoStuff_class_proportions_train = [0.0845890317, 0.0014816527, 0.0057424819, 0.0046404939, 0.0031459836, 0.0071818793, 0.0072533536, 0.0059259051, 0.0025378931, 0.0006547891, 0.0010705461, 0.0000000000, 0.0010705325, 0.0006622358, 0.0033758655, 0.0015221218, 0.0056149669, 0.0041308374, 0.0033977052, 0.0018115463, 0.0025429970, 0.0046248991, 0.0017546169, 0.0031238584, 0.0031381035, 0.0000000000, 0.0008658830, 0.0027966106, 0.0000000000, 0.0000000000, 0.0009794306, 0.0002680067, 0.0029893645, 0.0002948519, 0.0002937665, 0.0003422765, 0.0001275438, 0.0006371357, 0.0001364731, 0.0001761113, 0.0006013888, 0.0011386734, 0.0005501792, 0.0015023972, 0.0000000000, 0.0008324227, 0.0023678169, 0.0002659415, 0.0003969471, 0.0002317832, 0.0056967268, 0.0021250766, 0.0010162248, 0.0028081966, 0.0012708267, 0.0017130895, 0.0006748958, 0.0011987527, 0.0069771623, 0.0017099109, 0.0034590766, 0.0060452199, 0.0054128770, 0.0021972757, 0.0096239528, 0.0000000000, 0.0291459898, 0.0000000000, 0.0000000000, 0.0030160845, 0.0000000000, 0.0032271489, 0.0034435259, 0.0002093293, 0.0005159652, 0.0013403130, 0.0009161957, 0.0008753590, 0.0034513495, 0.0000594582, 0.0016835868, 0.0036071818, 0.0000000000, 0.0016771360, 0.0012382307, 0.0016206637, 0.0003334343, 0.0030314274, 0.0000323761, 0.0001495004, 0.0000000000, 0.0021910865, 0.0010841472, 0.0020302053, 0.0011507895, 0.0279449539, 0.0056294682, 0.0074368535, 0.0041874601, 0.0021503301, 0.0045194139, 0.0087119914, 0.0002741085, 0.0009450321, 0.0034672874, 0.0223945642, 0.0029930513, 0.0008989288, 0.0046125605, 0.0028649986, 0.0133893348, 0.0062665004, 0.0088211189, 0.0009333981, 0.0065451655, 0.0015300782, 0.0061474510, 0.0056212601, 0.0018797209, 0.0051245337, 0.0046619615, 0.0012044568, 0.0083807994, 0.0397281201, 0.0025830309, 0.0068588971, 0.0021717191, 0.0040828720, 0.0025989764, 0.0013233677, 0.0002361722, 0.0082389560, 0.0033562918, 0.0001723159, 0.0048287157, 0.0007966468, 0.0006212951, 0.0017536638, 0.0039155477, 0.0210479376, 0.0005179318, 0.0074976699, 0.0035636124, 0.0021275843, 0.0170166080, 0.0007837530, 0.0026964634, 0.0051307268, 0.0227824442, 0.0029929522, 0.0013826869, 0.0015675141, 0.0004887766, 0.0095368143, 0.0220318990, 0.0026647710, 0.0585924355, 0.0024254331, 0.0191723085, 0.0005162245, 0.0010596524, 0.0013679274, 0.0017867220, 0.0016255651, 0.0109771367, 0.0007981776, 0.0053053745, 0.0007388542, 0.0542156827, 0.0020164067, 0.0051984293, 0.0416266143, 0.0205099941, 0.0025627520, 0.0021155251, 0.0084735694, 0.0060459146, 0.0041491700, 0.0001213122, 0.0016519778, 0.0095374009, 0.0023980029]
cocoStuff_class_proportions_val = [0.0840889166, 0.0016792162, 0.0055210528, 0.0050689069, 0.0018382659, 0.0084995029, 0.0081340013, 0.0053383126, 0.0019845681, 0.0006087144, 0.0013675953, 0.0000000000, 0.0011160480, 0.0005637738, 0.0036522503, 0.0012049471, 0.0064835390, 0.0048930253, 0.0028015141, 0.0026601986, 0.0029181076, 0.0052023366, 0.0025356711, 0.0039642168, 0.0027244756, 0.0000000000, 0.0005676477, 0.0030591747, 0.0000000000, 0.0000000000, 0.0007265550, 0.0001364304, 0.0026598983, 0.0001957376, 0.0002130320, 0.0001499415, 0.0001119779, 0.0005490681, 0.0001356267, 0.0001906041, 0.0006481875, 0.0010801960, 0.0007243002, 0.0015948347, 0.0000000000, 0.0006211419, 0.0023587140, 0.0003648411, 0.0002873423, 0.0002686305, 0.0073164021, 0.0019552115, 0.0009843120, 0.0025200192, 0.0017983128, 0.0014826529, 0.0009088700, 0.0012555574, 0.0064867268, 0.0017027261, 0.0031119733, 0.0064692044, 0.0054985709, 0.0018468905, 0.0096294485, 0.0000000000, 0.0267100375, 0.0000000000, 0.0000000000, 0.0034074148, 0.0000000000, 0.0039840487, 0.0039916847, 0.0001995246, 0.0004898420, 0.0012832774, 0.0012921591, 0.0008173815, 0.0034673869, 0.0001114015, 0.0018347917, 0.0038531017, 0.0000000000, 0.0016952874, 0.0014824688, 0.0015602109, 0.0006330543, 0.0037154131, 0.0000513395, 0.0000631197, 0.0000000000, 0.0016219898, 0.0014377577, 0.0014658590, 0.0011508304, 0.0323970139, 0.0076690809, 0.0073383010, 0.0035631394, 0.0025123233, 0.0034938903, 0.0086767662, 0.0002694546, 0.0005879370, 0.0040483214, 0.0210727353, 0.0023365257, 0.0003821588, 0.0045645426, 0.0041727502, 0.0135094446, 0.0053901724, 0.0083931034, 0.0011707208, 0.0062567358, 0.0019560430, 0.0067409372, 0.0049847761, 0.0014551784, 0.0054977284, 0.0052552865, 0.0012637699, 0.0137802867, 0.0381918483, 0.0022717409, 0.0047581884, 0.0025303922, 0.0061950547, 0.0033461049, 0.0012215452, 0.0002524781, 0.0135243812, 0.0032953294, 0.0001601479, 0.0045686393, 0.0003620047, 0.0006569406, 0.0013182387, 0.0041473793, 0.0244803416, 0.0002767633, 0.0060377614, 0.0061766234, 0.0020202331, 0.0183564568, 0.0005518088, 0.0027950487, 0.0048363616, 0.0212030941, 0.0036148186, 0.0017932005, 0.0012513730, 0.0005311856, 0.0101196070, 0.0241129257, 0.0032538156, 0.0575602464, 0.0026598143, 0.0166345637, 0.0005761758, 0.0008626749, 0.0012158067, 0.0021163173, 0.0020722923, 0.0121962985, 0.0013976423, 0.0048416544, 0.0008094511, 0.0534586376, 0.0016507575, 0.0038204185, 0.0437774514, 0.0158912740, 0.0044937102, 0.0024431488, 0.0087617008, 0.0074032144, 0.0037155651, 0.0000392524, 0.0015924513, 0.0087014788, 0.0026670535]

cocoStuff_continuous_35_dict_deprecated = {
    0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6,
    9:7, 10:8,
    11:9, 12:9, # traffic sign
    13:10, 14:11, 63:12, 91:13,
    93:14, 96:14, 128:14, 141:14, 168:14, # vegetation
    95:15, 127:15, # building
    98:16, 99:17,
    110:18, 123:18, 124:18, 133:18, 135:18, 153:18, 158:18, # terrain
    112:19, 131:20, 139:21, 143:22,
    144:23, 145:24, 146:25, 148:26, 149:27, 150:28,
    160:29, 161:30, 163:31,
    170:32, 171:32, 172:32, 173:32, 174:32, 175:32, 176:32, # wall
    181:33, 255:34  # Background is mapped to 255, which is not used in the continuous labels.
}