"""
This script processes results of a model to calculate various metrics such as AUC-ROC score, precision, recall, specificity, F1-score and Precision-Recall AUC.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.express as px
from PIL import Image
import os
import glob
import re
import pandas as pd
import json

from utilities.post_process_file_utils import ResultsTest, common_target_classes, common_datasets, get_post_metrics
from utilities.main_file_utils import NumpyEncoder

results_test_mapillary_ios = ResultsTest(
    path_stem="results_test/model_final_mapillary_vistas_ios_point_mapper/*/*/*/*/*/",
    target_classes=common_target_classes,
    target_datasets=common_datasets,
    model_name="BiSeNetv2-iOS"
)
results_test_coco_ped = ResultsTest(
    path_stem="results_test/model_final_coco_combined_ios_point_mapper_finetuned/*/*/*/*/*/",
    target_classes=common_target_classes,
    target_datasets=common_datasets,
    model_name="BiSeNetv2-PED"
)
results_test_city = ResultsTest(
    path_stem="results_test/model_final_v2_city/*/*/*/*/*/",
    target_classes=common_target_classes,
    target_datasets=common_datasets,
    model_name="BiSeNetv2-City"
)
results = [results_test_mapillary_ios, results_test_coco_ped, results_test_city]

if __name__ == "__main__":
    for result in results:
        dataset_metrics = get_post_metrics(result)
        
        with open(os.path.join(result.output_dir, f"{result.model_name}_metrics.json"), 'w') as f:
            json.dump(dataset_metrics, f, indent=4, cls=NumpyEncoder)
        print(f"Metrics saved to {result.output_dir}/{result.model_name}_metrics.json")
        break