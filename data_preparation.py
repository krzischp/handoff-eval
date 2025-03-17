import pandas as pd
from evaluation import match_line_items
from similarity import llm_label_match_async
import asyncio

# from similarity import llm_label_match

line_item_info = [
    "label",
    "sectionName",
    "uom",
    "category",
    "qty",
    "rateUsd",
    "rowTotalCostUsd",
]


# Function to get GT vs Estimate dataframe
def get_gt_vs_estimate_for_model(gt_rows, pred_rows, matched_pairs):
    data = []
    for gt_idx, pred_idx in matched_pairs.items():
        gt_row = gt_rows[gt_idx]
        pred_row = pred_rows[pred_idx]
        entry = {f"gt_{key}": gt_row[key] for key in line_item_info}
        entry.update({f"pred_{key}": pred_row[key] for key in line_item_info})
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
    ✅ Computes recall: sum(df["similar_task"] == 1) / df.shape[0] for each example.
    ✅ Updates each example's 'matched_pairs_data' and stores recall.
    """
    matched_pairs = await add_llm_label_match_async(matched_pairs)

    # Compute recall
    for file_name in matched_pairs:
        df = matched_pairs[file_name]["matched_pairs_data"]
        matched_pairs[file_name]["recall"] = (
            sum(df["similar_task"] == 1) / df.shape[0] if df.shape[0] > 0 else 0
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

    # Step 1: Generate matched_pairs for all models
    for model_name, model_data in model_output_data.items():
        matched_pairs_dict[model_name] = match_line_item_pairs_for_model(
            model_data, ground_truth_data
        )

    # Step 2: Process all matched_pairs asynchronously
    tasks = {
        model_name: process_matched_pairs_async(matched_pairs)
        for model_name, matched_pairs in matched_pairs_dict.items()
    }

    processed_results = await asyncio.gather(*tasks.values())

    # Step 3: Update the dictionary with processed results
    for model_name, processed_data in zip(tasks.keys(), processed_results):
        matched_pairs_dict[model_name] = processed_data

    return matched_pairs_dict
