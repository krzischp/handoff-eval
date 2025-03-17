import os
import json
import numpy as np
import pandas as pd

from similarity import compute_similarity


# Function to load JSON files from a directory
def load_json_files(directory):
    data = {}
    for file in os.listdir(directory):
        if file.endswith(".json"):
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                data[file.replace(".json", "")] = json.load(f)
    return data


# Always picks the highest available similarity at each step
def match_line_items(gt_rows, pred_rows):
    if not gt_rows or not pred_rows:
        return {}

    cost_matrix = np.zeros((len(gt_rows), len(pred_rows)))

    # Compute similarity matrix
    for i, gt_row in enumerate(gt_rows):
        for j, pred_row in enumerate(pred_rows):
            cost_matrix[i, j] = compute_similarity(gt_row, pred_row)

    # Sort pairs by highest similarity score first
    sorted_pairs = sorted(
        [
            (i, j, cost_matrix[i, j])
            for i in range(len(gt_rows))
            for j in range(len(pred_rows))
        ],
        key=lambda x: -x[2],  # Sort by descending similarity
    )

    matched_pairs = {}
    used_gt = set()
    used_pred = set()

    # Greedily match the highest similarity pairs first
    for gt_idx, pred_idx, _ in sorted_pairs:
        if gt_idx not in used_gt and pred_idx not in used_pred:
            matched_pairs[gt_idx] = pred_idx
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)

    return matched_pairs


# Step 2: Compare Numerical Metrics (MAPE Calculation)
def calculate_mape(gt_rows, pred_rows, matched_pairs):
    qty_mape, rate_mape, cost_mape = [], [], []

    for gt_idx, pred_idx in matched_pairs.items():
        gt_row = gt_rows[gt_idx]
        pred_row = pred_rows[pred_idx]

        if gt_row["qty"] > 0:
            qty_mape.append(abs(gt_row["qty"] - pred_row["qty"]) / gt_row["qty"])
        if gt_row["rateUsd"] > 0:
            rate_mape.append(
                abs(gt_row["rateUsd"] - pred_row["rateUsd"]) / gt_row["rateUsd"]
            )
        if gt_row["rowTotalCostUsd"] > 0:
            cost_mape.append(
                abs(gt_row["rowTotalCostUsd"] - pred_row["rowTotalCostUsd"])
                / gt_row["rowTotalCostUsd"]
            )

    results = {
        "Qty MAPE": np.mean(qty_mape) if qty_mape else None,
        "Rate MAPE": np.mean(rate_mape) if rate_mape else None,
        "Cost MAPE": np.mean(cost_mape) if cost_mape else None,
    }
    return results


# Evaluate a Single Model
def evaluate_model(model_data, ground_truth_data):
    qty_mape, rate_mape, cost_mape = [], [], []

    for prediction in model_data["estimate_preds"]:
        file_name = prediction["valid_file_name"]
        if file_name not in ground_truth_data:
            continue

        gt_rows = ground_truth_data[file_name]["rows"]
        pred_rows = prediction["rows"]

        matched_pairs = match_line_items(gt_rows, pred_rows)
        mape_results = calculate_mape(gt_rows, pred_rows, matched_pairs)

        qty_mape.append(mape_results["Qty MAPE"])
        rate_mape.append(mape_results["Rate MAPE"])
        cost_mape.append(mape_results["Cost MAPE"])

    return {
        "Qty MAPE": np.mean([x for x in qty_mape if x is not None]),
        "Rate MAPE": np.mean([x for x in rate_mape if x is not None]),
        "Cost MAPE": np.mean([x for x in cost_mape if x is not None]),
    }


# Evaluate All Models Using the Modular Approach
def evaluate_all_models(model_output_data, ground_truth_data):
    model_results = {
        model_name: evaluate_model(model_data, ground_truth_data)
        for model_name, model_data in model_output_data.items()
    }
    return pd.DataFrame.from_dict(model_results, orient="index")
