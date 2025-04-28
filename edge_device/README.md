# Edge Device

This directory contains the edge-device friendly model files for the iOS Point Mapper app. The models are used to perform various tasks related to conversion of models developed in Pytorch to Executorch format.
The models are optimized for deployment on edge devices, ensuring efficient performance and reduced resource consumption.

## Semantic Segmentation

### BiSeNetv2

To perform Executorch conversion of the BiSeNetv2 model, the following steps were followed:
1. cd to the root directory of the project.
2. Run the following command to convert the model to Executorch format:

```bash
python edge_device/semantic_segmentation/bisenetv2.py \
    --weight-path <path_to_model_weights> \
    --im-size <input_image_size> \
    --outpath <output_path> \
    --img-path <test_input_image_path>
```
Here is an example command to convert the BiSeNetv2 model to Executorch format:
```bash
python edge_device/semantic_segmentation/bisenetv2.py \
    --weight-path ./model/semantic_segmentation/model_zoo/bisenetv2/model_final_v2_city.pth \
    --im-size 1024 512 \
    --outpath ./edge_device/semantic_segmentation/model_zoo \
    --img-path ./datasets/custom_images/test.jpg
```