import os
import glob
import pandas as pd
import re
import json

"""
Ad-hoc script.
Processes post_metrics.csv to return in desired format.

E.g. (Input)
Class ID,Class,Precision,Recall,F1-score,IoU,Pixel Count
1,sidewalk,0.9218118144477377,0.9014182028406009,0.9115009529873298,0.8373925135615851,5530027
2,building,0.7822509876054685,0.9336312225895519,0.8512634976814302,0.741043307985138,1391121
3,pole,0.5181371927681688,0.637609540285301,0.5716982931248211,0.4002643771781074,141044
4,traffic light,0.451505016722408,0.10196374622356495,0.16635859519408505,0.0907258064516129,2648
5,traffic sign,0.309175357111559,0.5749049429657794,0.4021047432707102,0.2516464965881262,18410

E.g. (Output)
Class, IoU, Precision, Recall

"""

csv_path_stem = [
    # "results_test/bisenetv2_*/*/*/*/*/*/post_metrics.csv",
    # "results_test/bisenetv2_*/*/*/*/*/*/*/post_metrics.csv"
    "results_test/model_final_*/*/*/*/*/*/post_metrics.csv",
]
# Example: results_test/bisenetv2_model_final_city/results_city_val/model_bisenetv2_city/split_val/s_2.0_sc_512_256/20250716-143444/post_metrics.csv
# Pattern: results_test/bisenetv2_<identifier1>/<identifier2>/model_<model_name>_<dataset_name>/split_<split_name>/s_<scale>_sc_<im_size>/YYYYMMDD-HHMMSS/post_metrics.csv
# Example: results_test/model_final_coco_combined_ios_point_mapper_finetuned/results_ios_point_mapper_val/model_bisenetv2_ios_point_mapper/split_val/s_2.0_sc_640_640/20250722-001237/post_metrics.csv

# Regex pattern to extract the model name from the path
# Example: results_test/bisenetv2_*/foo/bar/MODEL_NAME/...
identifiers_regrex_1 = re.compile(
    r"results_test/"
    r"model_final_(?P<identifier1>[^/]+)/"
    r"(?P<identifier2>[^/]+)/"
    r"model_(?P<model_name>[^_]+)_(?P<dataset_name>[^/]+)/"
    r"split_(?P<split_name>[^/]+)/"
    r"s_(?P<scale>[^_]+)_sc_(?P<im_size>[^/]+)/"
    r"(?P<timestamp>\d{8}-\d{6})/"
    r"post_metrics\.csv"
)
identifiers_regrex_2 = re.compile(
    r"results_test/"
    r"model_final_(?P<identifier1>[^/]+)/"
    r"(?P<identifier_2>[^/]+)/"
    r"(?P<identifier_null>[^/]+)/"
    r"model_(?P<model_name>[^_]+)_(?P<dataset_name>[^/]+)/"
    r"split_(?P<split_name>[^/]+)/"
    r"s_(?P<scale>[^_]+)_sc_(?P<im_size>[^/]+)/"
    r"(?P<timestamp>\d{8}-\d{6})/"
    r"post_metrics\.csv"
)
identifiers_regex_list = [identifiers_regrex_1, identifiers_regrex_2]
# print(f"Identifiers regex: {identifiers_regrex.pattern}")

def process_post_metrics_csv(csv_path):
    """
    Processes the post_metrics.csv file to return a DataFrame with the desired columns.
    
    Args:
        csv_path (str): Path to the post_metrics.csv file.
        
    Returns:
        pd.DataFrame: Processed DataFrame with selected columns.
    """
    # Get the model details from the csv_path
    model_details = None
    for regex in identifiers_regex_list:
        match = regex.match(csv_path)
        if match:
            model_details = match.groupdict()
            break
    
    processed_df = pd.DataFrame(columns=[
        "Model Name", "Model Identifier", "Dataset Name", "Class ID", "Class Name", "IoU", "Precision", "Recall", "F1-score"
    ])
    
    # Read the csv file and get the required metrics
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Warning: The CSV file at {csv_path} is empty.")
        return pd.DataFrame()
    
    new_df = df[[
        "Class ID", "Class", "IoU", "Precision", "Recall", "F1-score"
    ]].rename(columns={
        "Class": "Class",
        "IoU": "IoU",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1-score": "F1-score"
    })
    
    # Reduce precision of float columns
    float_columns = ["IoU", "Precision", "Recall", "F1-score"]
    for col in float_columns:
        if col in new_df.columns:
            new_df[col] = new_df[col].round(6)
    
    if model_details:
        new_df.insert(0, "Model Name", model_details["model_name"])
        new_df.insert(1, "Model Identifier", model_details["identifier1"])
        new_df.insert(2, "Dataset Name", model_details["dataset_name"])
    else:
        print(f"Warning: Could not extract model details from the path {csv_path}.")
        new_df.insert(0, "Model Name", "Unknown")
        new_df.insert(1, "Model Identifier", "Unknown")
        new_df.insert(2, "Dataset Name", "Unknown")
    
    return new_df

csv_paths = []
csv_path_details = []
for stem in csv_path_stem:
    found_paths = glob.glob(stem)
    csv_paths.extend(found_paths)

print(f"Found {len(csv_paths)} post_metrics.csv files")

metrics_df = pd.DataFrame()
for csv_path in csv_paths:
    df = process_post_metrics_csv(csv_path)
    metrics_df = pd.concat([metrics_df, df], ignore_index=True)

output_csv_path = "results_test/processed_post_metrics.csv"
metrics_df.to_csv(output_csv_path, index=False)
print(f"Processed metrics saved to {output_csv_path}")
