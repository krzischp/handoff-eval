import asyncio
import json
import os

import pandas as pd

from .constants import LINE_ITEM_INFO
from .logger import logger
from .similarity import llm_label_match_async, match_line_items


# Function to load JSON files from a directory
def load_json_files(directory):
    data = {}
    for file in os.listdir(directory):
        if file.endswith(".json"):
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                data[file.replace(".json", "")] = json.load(f)
    return data


# Function to get GT vs Estimate dataframe
def get_gt_vs_estimate_for_model(gt_rows, pred_rows, matched_pairs):
    data = []
    for gt_idx, pred_idx in matched_pairs.items():
        gt_row = gt_rows[gt_idx]
        pred_row = pred_rows[pred_idx]
        entry = {f"gt_{key}": gt_row[key] for key in LINE_ITEM_INFO}
        entry.update({f"pred_{key}": pred_row[key] for key in LINE_ITEM_INFO})
        data.append(entry)
    return pd.DataFrame(data)


# Function to calculate similarity data for a single model
def match_line_item_pairs_for_model(model_data, ground_truth_data):
    all_matched_pairs = {}
    for prediction in model_data["estimate_preds"]:
        file_name = prediction["valid_file_name"]
        if file_name not in ground_truth_data:
            continue

        gt_rows = ground_truth_data[file_name]["rows"]
        pred_rows = prediction["rows"]
        matched_pairs = match_line_items(gt_rows, pred_rows)
        all_matched_pairs[file_name] = dict()
        all_matched_pairs[file_name]["matched_pairs"] = matched_pairs
        matched_pairs_data = get_gt_vs_estimate_for_model(
            gt_rows, pred_rows, matched_pairs
        )
        all_matched_pairs[file_name]["matched_pairs_data"] = matched_pairs_data
        all_matched_pairs[file_name]["n_estimations"] = len(pred_rows)

    return all_matched_pairs


# -------------- ASYNC ----------------
async def process_row(row):
    return await llm_label_match_async(row["gt_label"], row["pred_label"])


async def add_llm_label_match_to_df_async(df):
    """
    Enhances a single DataFrame with LLM-based similarity evaluation asynchronously.

    ✅ Adds 'similar_task' (1 if similar, 0 otherwise)
    ✅ Adds 'justification' (LLM reasoning for the decision)
    """
    tasks = [process_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    df[["similar_task", "justification"]] = pd.DataFrame(results)
    return df


async def add_llm_label_match_async(matched_pairs):
    """
    Iterates over matched_pairs and applies LLM-based label matching to each file asynchronously.
    """
    tasks = {}
    for file_name, data in matched_pairs.items():
        tasks[file_name] = add_llm_label_match_to_df_async(data["matched_pairs_data"])

    updated_dataframes = await asyncio.gather(*tasks.values())

    # Update matched_pairs structure with processed data
    for file_name, df in zip(tasks.keys(), updated_dataframes):
        matched_pairs[file_name]["matched_pairs_data"] = df

    return matched_pairs


async def process_matched_pairs_async(matched_pairs):
    """
    Enhances matched_pairs with LLM-based similarity evaluation and computes recall asynchronously.

    ✅ Applies LLM-based label matching to add 'similar_task' and 'justification' columns.
    ✅ Computes recall, precision, f1
    ✅ Updates each example's 'matched_pairs_data' and stores recall.
    """
    matched_pairs = await add_llm_label_match_async(matched_pairs)

    # Compute classification metrics
    for file_name in matched_pairs:
        df = matched_pairs[file_name]["matched_pairs_data"]
        n_similar = sum(df["similar_task"] == 1)
        n_estimations = matched_pairs[file_name]["n_estimations"]
        n_gt = df.shape[0]
        recall = n_similar / n_gt if n_gt > 0 else 0
        precision = n_similar / n_estimations if n_estimations > 0 else 0
        matched_pairs[file_name]["recall"] = recall
        matched_pairs[file_name]["precision"] = precision
        matched_pairs[file_name]["f1"] = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    return matched_pairs


async def process_all_models_async(model_output_data, ground_truth_data):
    """
    Iterates over all models in model_output_data, computes matched_pairs,
    processes them asynchronously, and stores results in a dictionary.

    Returns:
        dict[model_name] -> processed matched_pairs (including recall and LLM similarity)
    """
    matched_pairs_dict = {}

    logger.info("Step 1: Generating matched_pairs for all models.")
    for model_name, model_data in model_output_data.items():
        matched_pairs_dict[model_name] = match_line_item_pairs_for_model(
            model_data, ground_truth_data
        )

    logger.info("Step 2: Processing all matched_pairs asynchronously.")
    tasks = {
        model_name: process_matched_pairs_async(matched_pairs)
        for model_name, matched_pairs in matched_pairs_dict.items()
    }

    processed_results = await asyncio.gather(*tasks.values())

    logger.info("Step 3: Updating matched_pairs_dict with processed results.")
    for model_name, processed_data in zip(tasks.keys(), processed_results):
        matched_pairs_dict[model_name] = processed_data

    logger.info("Processing complete.")
    return matched_pairs_dict
