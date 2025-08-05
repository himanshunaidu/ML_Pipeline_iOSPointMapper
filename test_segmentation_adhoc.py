"""
This script processes results of a model to calculate various metrics such as AUC-ROC score, precision, recall, specificity, F1-score and Precision-Recall AUC.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
import os
import glob
import re
import pandas as pd
import json

from utilities.post_process_file_utils import ResultsTest, common_target_classes, common_datasets, get_post_metrics, \
    create_csv_from_metrics, \
    get_radar_graph_for_classes, get_radar_graph_for_metrics_per_class, \
    get_auc_roc_pr_scores, get_auc_roc_curves, get_pr_curves, \
    save_fuse_segmentation_images, save_entropy_maps
from utilities.main_file_utils import NumpyEncoder, NumpyDecoder

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
results_test = [results_test_mapillary_ios, results_test_coco_ped, results_test_city]

if __name__ == "__main__":
    model_results = {result_args.model_name: None for result_args in results_test}
    # model_results_csv = pd.DataFrame()
    
    # for result_args in results_test:
    #     if not os.path.exists(result_args.output_dir):
    #         os.makedirs(result_args.output_dir)
    #     if os.path.exists(os.path.join(result_args.output_dir, f"{result_args.model_name}_metrics.json")):
    #         print(f"Metrics file already exists for {result_args.model_name}. Skipping...")
    #     else:
    #         dataset_metrics = get_post_metrics(result_args)
    #         with open(os.path.join(result_args.output_dir, f"{result_args.model_name}_metrics.json"), 'w') as f:
    #             json.dump(dataset_metrics, f, indent=4, cls=NumpyEncoder)
    #         print(f"Metrics saved to {result_args.output_dir}/{result_args.model_name}_metrics.json")
            
    #     with open(os.path.join(result_args.output_dir, f"{result_args.model_name}_metrics.json"), 'r') as f:
    #         model_results[result_args.model_name] = json.load(f, cls=NumpyDecoder)
            
        # results_csv = create_csv_from_metrics(model_results[result_args.model_name], output_path=result_args.output_dir)
        # model_results_csv = pd.concat([model_results_csv, results_csv], ignore_index=True)
        
    # model_results_csv.to_csv(os.path.join(result_args.output_dir, f"metrics.csv"), index=False)
    # print(f"Model results created for {len(model_results)} models.")
    
    # Generate Radar graph for iou_score, with all the models and classes, for dataset 'Combined'
    # target_dataset = ("Combined", ["results_edge_mapping_ios_val", "results_ios_point_mapper_val"])
    # target_classes = common_target_classes
    # radar_graph = get_radar_graph_for_classes(model_results, target_classes, target_dataset, "iou_score")
    # radar_graph.write_html("radar_graph.html")
    
    # Generate Radar graph for every class in the dataset 'Combined'
    # target_dataset = ("Combined", ["results_edge_mapping_ios_val", "results_ios_point_mapper_val"])
    # target_classes = common_target_classes
    # if not os.path.exists(os.path.join(results_test_mapillary_ios.output_dir, "per_class")):
    #     os.makedirs(os.path.join(results_test_mapillary_ios.output_dir, "per_class"))
    # for class_name, class_id in target_classes.items():
    #     radar_graph = get_radar_graph_for_metrics_per_class(
    #         model_results, target_class=(class_name, class_id), 
    #         target_dataset=target_dataset)
    #     if radar_graph:
    #         radar_graph.write_html(
    #             os.path.join(results_test_mapillary_ios.output_dir, "per_class", f"{class_name}_radar_graph.html")
    #         )
    #         print(f"Radar graph for {class_name} saved.")
    
    # Generate AUC-ROC and Precision-Recall graphs for each class across all models
    # target_dataset = ("Combined", ["results_edge_mapping_ios_val", "results_ios_point_mapper_val"])
    # target_classes = common_target_classes
    
    # model_results_for_plot = {result_args.model_name: None for result_args in results_test}
    # for result_args in results_test:
    #     model_name = result_args.model_name
    #     if os.path.exists(os.path.join(result_args.output_dir, f"{model_name}_metrics_for_plot.json")):
    #         print(f"Metrics for plot already exists for {model_name}. Skipping...")
    #     else:
    #         model_results_for_plot[model_name] = get_auc_roc_pr_scores(result_args, target_dataset, target_classes, subset_step=100000)
    #         with open(os.path.join(result_args.output_dir, f"{model_name}_metrics_for_plot.json"), 'w') as f:
    #             json.dump(model_results_for_plot[model_name], f, indent=4, cls=NumpyEncoder)
        
    #     with open(os.path.join(result_args.output_dir, f"{model_name}_metrics_for_plot.json"), 'r') as f:
    #         model_results_for_plot[model_name] = json.load(f, cls=NumpyDecoder)
    
    # if not os.path.exists(os.path.join(results_test_mapillary_ios.output_dir, "auc_roc_curves")):
    #     os.makedirs(os.path.join(results_test_mapillary_ios.output_dir, "auc_roc_curves"))
    # if not os.path.exists(os.path.join(results_test_mapillary_ios.output_dir, "pr_curves")):
    #     os.makedirs(os.path.join(results_test_mapillary_ios.output_dir, "pr_curves"))
    # for class_name, class_id in target_classes.items():
    #     target_class = (class_name, class_id)
    #     fig_auc_roc = get_auc_roc_curves(model_results_for_plot, target_dataset, target_class)
    #     if fig_auc_roc:
    #         fig_auc_roc.write_html(
    #             os.path.join(results_test_mapillary_ios.output_dir, "auc_roc_curves", f"{class_name}_auc_roc_curves.html")
    #         )
    #         print(f"AUC-ROC curves for {class_name} saved.")
    #     fig_pr = get_pr_curves(model_results_for_plot, target_dataset, target_class)
    #     if fig_pr:
    #         fig_pr.write_html(
    #             os.path.join(results_test_mapillary_ios.output_dir, "pr_curves", f"{class_name}_pr_curves.html")
    #         )
    #         print(f"Precision-Recall curves for {class_name} saved.")
    # print("All AUC-ROC and Precision-Recall curves generated.")
    
    # results_test_list = results_test[::-1]
    # save_fuse_segmentation_images(
    #     results_test_list=results_test_list,
    #     target_dataset=("iOS", ["results_ios_point_mapper_val"]),
    #     output_dir=os.path.join(results_test_mapillary_ios.output_dir, 'fuse_segmentation_images')
    # )
    
    # Save the entropy maps for each model
    # target_dataset = ("Combined", ["results_edge_mapping_ios_val", "results_ios_point_mapper_val"])
    target_dataset = ("iOS", ["results_ios_point_mapper_val"])
    # for result_args in results_test:
    entropy_output_dir = os.path.join(results_test_mapillary_ios.output_dir, 'entropy_maps')
    if not os.path.exists(entropy_output_dir):
        os.makedirs(entropy_output_dir)
    save_entropy_maps(results_test_mapillary_ios, target_dataset, 
        output_dir=entropy_output_dir
    )