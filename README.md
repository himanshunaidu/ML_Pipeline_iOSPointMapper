# CoreML Pipeline iOSPointMapper

This is a Machine Learning pipeline for the iOSPointMapper application that deals with training (predominantly vision) models and converting them to CoreML format. 
This project takes heavy inspiration from [EdgeNets](https://github.com/sacmehta/EdgeNets.git), but several modifications are made to work with newer library versions. 

At the moment, the pipeline is designed to help in conversion of models from PyTorch to CoreML, Executorch or other formats, as well as consistent inference and evaluation of models.

## Train (Incomplete)

The model, while attempting to integrate training as well, is not fully functional yet, and there may not be much support for it. This is mostly due to the diversity of model training pipelines, including differences in the data loading, augmentation, training loops, and other training-related components.

Hence, the `train_segmentation.py` script and the `train` folder are currently only for reference, and it is not recommended to use them for training. Instead, it is recommended to perform the training in the original repositories, and then bring the model to this repository for conversion and inference.