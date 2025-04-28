# CoreML

This directory contains the CoreML model files for the iOS Point Mapper app. The models are used to perform various tasks related to conversion of models developed in Pytorch to CoreML format.

## Semantic Segmentation

### BiSeNetv2

To perform CoreML conversion of the BiSeNetv2 model, the following steps were followed:
1. cd to the root directory of the project.

2. Run the following command to convert the model to CoreML format:

```bash
python coreml/semantic_segmentation/bisenetv2.py \
    --weight-path <path_to_model_weights> \
    --im-size <input_image_size> \
    --outpath <output_path> \
    --img-path <test_input_image_path>
```

Here is an example command to convert the BiSeNetv2 model to CoreML format:
```bash
python coreml/semantic_segmentation/bisenetv2.py \
    --weight-path ./model/semantic_segmentation/model_zoo/bisenetv2/model_final_v2_city.pth \
    --im-size 1024 512 \
    --outpath ./coreml/semantic_segmentation/model_zoo \
    --img-path ./datasets/custom_images/test.jpg
```

### ESPNetv2

To perform CoreML conversion of the ESPNetv2 model, the following steps were followed:
1. cd to the root directory of the project.
2. Run the following command to convert the model to CoreML format:

```bash
python coreml/semantic_segmentation/espnetv2.py \
    --weight-path <path_to_model_weights> \
    --s <scale> \
    --im-size <input_image_size> \
    --outpath <output_path> \
    --img-path <test_input_image_path>
    --dataset <dataset_name> \
    --data-path <dataset_path>
```

Here is an example command to convert the ESPNetv2 model to CoreML format:
```bash
python coreml/semantic_segmentation/espnetv2.py \
    --weight-path ./model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth \
    --im-size 1024 512 \
    --s 2.0 \
    --outpath ./coreml/semantic_segmentation/model_zoo \
    --img-path ./datasets/custom_images/test.jpg \
    --dataset city \
    --data-path ./datasets/cityscapes
```