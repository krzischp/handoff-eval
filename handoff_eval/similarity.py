import re

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from .constants import FUZZY_MATCH_THRESHOLD, STOPWORDS
from .openai_client import DEFAULT_MODEL, call_openai_with_limit


def preprocess_label(label):
    """Lowercase and remove common stopwords to extract meaningful keywords."""
    words = re.findall(r"\b\w+\b", label.lower())  # Extract words
    return set(word for word in words if word not in STOPWORDS)


def fuzzy_keyword_match(gt_keywords, pred_keywords, threshold=FUZZY_MATCH_THRESHOLD):
    """
    Computes fuzzy matches between keyword sets.
    ✅ Matches similar words (e.g., "install" ~ "instal").
    ✅ Uses a threshold to consider words as similar.
    """
    matched = 0
    for gt_word in gt_keywords:
        for pred_word in pred_keywords:
            similarity = fuzz.ratio(gt_word, pred_word)
            # Consider as a match if similarity is high
            if similarity >= threshold:
                matched += 1
                # Avoid double counting
                break

    return matched


def compute_keyword_overlap_score(
    gt_label, pred_label, fuzzy=True, threshold=FUZZY_MATCH_THRESHOLD
):
    """
    Computes a similarity score based on common keywords between GT and estimated labels.
    ✅ Uses Jaccard-like similarity with fuzzy matching for typos.
    ✅ Adjusts overlap based on a similarity threshold.
    """

    gt_keywords = preprocess_label(gt_label)
    pred_keywords = preprocess_label(pred_label)

    if not gt_keywords or not pred_keywords:
        return 0  # No meaningful words to compare

    if fuzzy:
        intersection = fuzzy_keyword_match(gt_keywords, pred_keywords, threshold)
    else:
        intersection = len(gt_keywords & pred_keywords)  # Exact match comparison

    union = len(gt_keywords | pred_keywords)  # Total unique keywords

    # Compute similarity score
    similarity_score = intersection / union if union > 0 else 0

    return similarity_score


def min_max_normalize(value, min_val, max_val):
    """
    Normalize a value to the range [0, 1] using min-max scaling.
    Prevents division by zero by ensuring (max_val - min_val) is not zero.
    """
    if max_val - min_val == 0:
        return 0  # Avoid division by zero edge case

    return (value - min_val) / (max_val - min_val)


def compute_similarity(gt_row, pred_row, fuzzy_threshold=FUZZY_MATCH_THRESHOLD):
    """Compute total similarity score using keyword overlap & binary fuzzy match, handling NaN values."""

    # Handle NaN values by replacing with empty strings
    gt_label = gt_row["label"] if pd.notna(gt_row["label"]) else ""
    pred_label = pred_row["label"] if pd.notna(pred_row["label"]) else ""

    gt_section = gt_row["sectionName"] if pd.notna(gt_row["sectionName"]) else ""
    pred_section = pred_row["sectionName"] if pd.notna(pred_row["sectionName"]) else ""

    gt_category = gt_row["category"] if pd.notna(gt_row["category"]) else ""
    pred_category = pred_row["category"] if pd.notna(pred_row["category"]) else ""

    gt_uom = gt_row["uom"] if pd.notna(gt_row["uom"]) else ""
    pred_uom = pred_row["uom"] if pd.notna(pred_row["uom"]) else ""

    # 1️⃣ Compute keyword-based similarity (Jaccard-like score in [0,1])
    label_similarity = compute_keyword_overlap_score(gt_label, pred_label)

    # 2️⃣ Compute fuzzy matches (converted to binary 1/0)
    section_match = int(
        fuzz.token_sort_ratio(gt_section, pred_section) >= fuzzy_threshold
    )
    category_match = int(
        fuzz.token_sort_ratio(gt_category, pred_category) >= fuzzy_threshold
    )
    uom_match = int(fuzz.token_sort_ratio(gt_uom, pred_uom) >= fuzzy_threshold)

    # Weight of 0 or 1 for each structured field and a score in range 0-1 on label to discriminate similar matches
    total_score = (
        label_similarity  # Label similarity (task understanding)
        + uom_match  # UOM is critical
        + category_match  # Category affects work type
        + section_match  # Section errors matter but are less critical
    )

    return total_score


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


async def llm_label_match_async(gt_label, pred_label, model=DEFAULT_MODEL):
    system_prompt = (
        "You are a highly experienced construction estimator specializing in residential projects. "
        "Your task is to determine whether two given labels describe a related construction task, "
        "even if they are worded differently. Accuracy is critical because incorrect matches can lead to costly estimation errors. "
        "Use your domain knowledge to account for synonyms, variations in phrasing, and common construction terminology."
    )

    user_prompt = (
        f"Do the following two labels refer to a related construction task?\n\n"
        f"1. {gt_label}\n"
        f"2. {pred_label}\n\n"
        f"Consider whether these labels describe the same type of work, even if phrased differently. "
        f"Respond with 'Yes' or 'No' followed by a brief justification of one sentence max."
    )

    response = await call_openai_with_limit(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    message = response.choices[0].message.content.lower()
    return {"similar_task": 1 if "yes" in message else 0, "justification": message}
