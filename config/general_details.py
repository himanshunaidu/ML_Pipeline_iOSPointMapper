# classification related details
# classification_datasets = ['imagenet', 'coco']
# classification_schedulers = ['fixed', 'clr', 'hybrid', 'linear', 'poly']
# classification_models = ['espnetv2']
# classification_exp_choices = ['main', 'ablation']

# segmentation related details
segmentation_schedulers = ['poly', 'fixed', 'clr', 'linear', 'hybrid']
segmentation_datasets = ['pascal', 'city', 'edge_mapping', 'coco_stuff', 'ios_point_mapper']
segmentation_models = ['espnetv2', 'bisenetv2']
segmentation_loss_fns = ['ce', 'bce'] # OHEM for BiSeNetV2, but is currently not implemented


# detection related details

# detection_datasets = ['coco', 'pascal']
# detection_models = ['espnetv2', 'dicenet']
# detection_schedulers = ['poly', 'hybrid', 'clr', 'cosine']
