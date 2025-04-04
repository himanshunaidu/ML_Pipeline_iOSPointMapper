model_weight_map = {}
# key is of the form <model-name_model-scale>

## ESPNetv2 models
espnetv2_scales = [0.5, 1.0, 1.25, 1.5, 2.0]
for sc in espnetv2_scales:
    model_weight_map['espnetv2_{}'.format(sc)] = 'model/semantic_segmentation/espnetv2/classification_submodule/model_zoo/espnetv2_s_{}_imagenet_224x224.pth'.format(sc)
