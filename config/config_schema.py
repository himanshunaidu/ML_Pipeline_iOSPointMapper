# config_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional


class TestConfig(BaseModel):
    # General details
    workers: int = Field(default=4, description='Number of data loading workers')

    # Model details
    model: str = Field(default='bisenetv2', description='Model name')
    weights_test: str = Field(default='', description='Pretrained weights path')
    s: float = Field(default=2.0, description='Scale parameter')

    # Dataset details
    dataset: str = Field(default='city', description='Dataset name')
    data_path: str = Field(default='', description='Path to dataset')

    # Input details
    im_size: List[int] = Field(default_factory=lambda: [512, 256], description='Input image size [W, H]')
    split: str = Field(default='val', description='Dataset split: train/val/test')
    batch_size: int = Field(default=1, description='Batch size')
    model_width: int = Field(default=224)
    model_height: int = Field(default=224)
    channels: int = Field(default=3)

    is_custom: bool = Field(default=False, description='Whether to use a custom label mapping')
    custom_mapping_dict_key: Optional[str] = Field(default=None, description='Key for custom mapping dictionary')

    # Output details
    num_classes: int = Field(default=19, description='Number of output classes')
    savedir: str = Field(default='./results_segmentation_test', description='Save directory for results')
    
    # Weights, classes, mean and std
    weights: Optional[str] = Field(default='', description='Path to weights for training')
    classes: Optional[int] = Field(default=None, description='Number of classes for the model')
    mean: Optional[List[float]] = Field(default=None, description='Mean values for normalization')
    std: Optional[List[float]] = Field(default=None, description='Standard deviation values for normalization')
