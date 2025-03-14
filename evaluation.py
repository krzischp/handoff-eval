import numpy as np
import pandas as pd
import os
import json


# Function to evaluate a single model variant
def evaluate_model(model_predictions, ground_truth_data):
    """
    Evaluates a single model variant against ground truth.
    """
    line_item_precision, line_item_recall, line_item_f1 = [], [], []
    qty_mape, rate_mape, cost_mape = [], [], []
    qty_rmse, rate_rmse, cost_rmse = [], [], []
    total_cost_mape, total_cost_rmse = [], []

    # Iterate over test cases in the model predictions
    for prediction in model_predictions["estimate_preds"]:
        file_name = prediction["valid_file_name"]
        if file_name not in ground_truth_data:
            continue

        # Ground truth
        gt_rows = ground_truth_data[file_name]["rows"]
        gt_total_cost = ground_truth_data[file_name]["totalCostUsd"]

        # Model outputs (average over 2 estimations)
        preds_list = prediction["rows"]

        # Convert to DataFrames for easier comparison
        gt_df = pd.DataFrame(gt_rows)
        pred_df = pd.DataFrame(preds_list)

        # Line-item matching
        gt_labels = set(gt_df["label"])
        pred_labels = set(pred_df["label"])
        true_positives = len(gt_labels & pred_labels)
        precision = true_positives / len(pred_labels) if len(pred_labels) > 0 else 0
        recall = true_positives / len(gt_labels) if len(gt_labels) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        line_item_precision.append(precision)
        line_item_recall.append(recall)
        line_item_f1.append(f1)

        # Quantity, Rate, and Cost Errors
        merged_df = gt_df.merge(
            pred_df, on="label", suffixes=("_gt", "_pred"), how="inner"
        )

        if not merged_df.empty:
            qty_mape.append(
                np.mean(
                    np.abs(
                        (merged_df["qty_gt"] - merged_df["qty_pred"])
                        / merged_df["qty_gt"]
                    )
                )
            )
            rate_mape.append(
                np.mean(
                    np.abs(
                        (merged_df["rateUsd_gt"] - merged_df["rateUsd_pred"])
                        / merged_df["rateUsd_gt"]
                    )
                )
            )
            cost_mape.append(
                np.mean(
                    np.abs(
                        (
                            merged_df["rowTotalCostUsd_gt"]
                            - merged_df["rowTotalCostUsd_pred"]
                        )
                        / merged_df["rowTotalCostUsd_gt"]
                    )
                )
            )

            qty_rmse.append(
                np.sqrt(np.mean((merged_df["qty_gt"] - merged_df["qty_pred"]) ** 2))
            )
            rate_rmse.append(
                np.sqrt(
                    np.mean((merged_df["rateUsd_gt"] - merged_df["rateUsd_pred"]) ** 2)
                )
            )
            cost_rmse.append(
                np.sqrt(
                    np.mean(
                        (
                            merged_df["rowTotalCostUsd_gt"]
                            - merged_df["rowTotalCostUsd_pred"]
                        )
                        ** 2
                    )
                )
            )

        # Total Cost Accuracy
        pred_total_cost = pred_df["rowTotalCostUsd"].sum()
        total_cost_mape.append(abs(gt_total_cost - pred_total_cost) / gt_total_cost)
        total_cost_rmse.append((gt_total_cost - pred_total_cost) ** 2)

    # Aggregate results
    results = {
        "Precision": np.mean(line_item_precision),
        "Recall": np.mean(line_item_recall),
        "F1 Score": np.mean(line_item_f1),
        "Qty MAPE": np.mean(qty_mape),
        "Rate MAPE": np.mean(rate_mape),
        "Cost MAPE": np.mean(cost_mape),
        "Qty RMSE": np.mean(qty_rmse),
        "Rate RMSE": np.mean(rate_rmse),
        "Cost RMSE": np.mean(cost_rmse),
        "Total Cost MAPE": np.mean(total_cost_mape),
        "Total Cost RMSE": np.sqrt(np.mean(total_cost_rmse)),
    }
    return results


def load_json_files(directory):
    """
    Loads all JSON files from a given directory into a dictionary.

    Args:
        directory (str): Path to the directory containing JSON files.

    Returns:
        dict: A dictionary with file names as keys and JSON content as values.
    """
    data = {}
    for file in os.listdir(directory):
        if file.endswith(".json"):
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                data[file.replace(".json", "")] = json.load(f)
    return data


def evaluate_all_models(model_output_data, ground_truth_data):
    """
    Evaluates all model variants against the ground truth.

    Args:
        model_output_data (dict): Dictionary containing model output JSONs.
        ground_truth_data (dict): Dictionary containing ground truth JSONs.

    Returns:
        pd.DataFrame: A dataframe with evaluation results for each model.
    """
    model_results = {
        model_name: evaluate_model(model_data, ground_truth_data)
        for model_name, model_data in model_output_data.items()
    }
    return pd.DataFrame.from_dict(model_results, orient="index").sort_index(
        key=lambda x: x.astype(int)
    )
