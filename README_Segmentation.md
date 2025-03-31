# Segmentation Code

## Train

We train our segmentation networks in two stages:
- In the first stage, we use a low-resolution image as an input so that we can fit a larger batch-size. See below command
```bash
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset city --data-path ./datasets/cityscapes/ --batch-size 25 --crop-size 512 256 --lr 0.009 --scheduler hybrid --clr-max 61 --epochs 100
```

- In the second stage, we freeze the batch normalization layers and then finetune at a slightly higher image resolution. See below command (replace finetune path with your the best model path from the first stage):
```bash
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset city --data-path ./datasets/cityscapes/ --batch-size 6 --crop-size 1024 512 --lr 0.005 --scheduler hybrid --clr-max 61 --epochs 100 --freeze-bn --finetune ./results_segmentation/model_espnetv2_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5/20250331-071431/espnetv2_2.0_512_best.pth
```

The following is an example of VSCode launch config:
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "iOSPointMapper: Train Model Stage 1",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/ubuntu/CoreML_Pipeline_iOSPointMapper/train_segmentation.py",
            "cwd": "/home/ubuntu/CoreML_Pipeline_iOSPointMapper",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
            },
            //--model espnetv2 --s 2.0 --dataset city --data-path ./datasets/cityscapes/ --batch-size 25 --crop-size 512 256 --lr 0.009 --scheduler hybrid --clr-max 61 --epochs 100
            "args": [
                "--model", "espnetv2",
                "--s", "2.0",
                "--dataset", "city",
                "--data-path", "./datasets/cityscapes/",
                "--batch-size", "25",
                "--crop-size", "512", "256",
                "--lr", "0.009",
                "--scheduler", "hybrid",
                "--clr-max", "61",
                "--epochs", "100"
            ]
        },
        {
            "name": "iOSPointMapper: Train Model Stage 2",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/ubuntu/CoreML_Pipeline_iOSPointMapper/train_segmentation.py",
            "cwd": "/home/ubuntu/CoreML_Pipeline_iOSPointMapper",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
            },
            //--model espnetv2 --s 2.0 --dataset city --data-path ./datasets/cityscapes/ --batch-size 6 --crop-size 1024 512 --lr 0.005 --scheduler hybrid --clr-max 61 --epochs 100 --freeze-bn --finetune model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth
            "args": [
                "--model", "espnetv2",
                "--s", "2.0",
                "--dataset", "city",
                "--data-path", "./datasets/cityscapes/",
                "--batch-size", "6",
                "--crop-size", "1024", "512",
                "--lr", "0.005",
                "--scheduler", "hybrid",
                "--clr-max", "61",
                "--epochs", "100",
                "--freeze-bn",
                "--finetune", "model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth"
            ]
        }
    ]
}
```

## Test

For running Cityscapes dataset, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset city --data-path ./datasets/cityscapes/ --split val --im-size 1024 512 --weights-test model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth
```

For running Edge_Mapping dataset, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset edge_mapping --data-path ./datasets/edge_mapping/NorthSeattle_1118 --split val --im-size 1024 512 --weights-test model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth
```