model_weight_map = {}
# key is of the form <model-name_model-scale>


#ESPNetv2
espnetv2_scales = [0.5, 1.0, 1.5, 2.0]
for scale in espnetv2_scales:
    model_weight_map['espnetv2_{}'.format(scale)] = {
        'pascal_256x256':
            {
                'weights': 'model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_{}_pascal_256x256.pth'.format(scale)
            },
        'pascal_384x384':
            {
                'weights': 'model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_{}_pascal_384x384.pth'.format(scale)
            },
        'city_1024x512': {
            'weights': 'model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_{}_city_1024x512.pth'.format(scale)
        },
        'city_512x256': {
            'weights': 'model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_{}_city_512x256.pth'.format(scale)
        }
    }
