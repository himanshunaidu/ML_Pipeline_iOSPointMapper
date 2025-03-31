# Segmentation Code

## Test

For running Cityscapes dataset, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset city --data-path ./datasets/cityscapes/ --split val --im-size 1024 512 --weights-test model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth
```

For running Edge_Mapping dataset, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset edge_mapping --data-path ./datasets/edge_mapping/NorthSeattle_1118 --split val --im-size 1024 512 --weights-test model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth
```