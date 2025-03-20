import numpy as np
import pandas as pd


def compute_word_count_buckets(ground_truth_data, n_buckets=3):
    """
    Computes bucket edges for word counts based on the full dataset.

    Parameters:
    - ground_truth_data (dict): Dictionary containing ground truth data with "input" texts.
    - n_buckets (int): The number of buckets to categorize the word count.

    Returns:
    - list: The bucket edges to be used for assigning bucket indices.
    """
    # Extract word counts from all examples
    word_counts = [len(data["input"].split()) for data in ground_truth_data.values()]

    # Define bucket edges based on percentiles
    bucket_edges = np.percentile(word_counts, np.linspace(0, 100, n_buckets + 1)[1:-1])

    return bucket_edges


def label_buckets(bucket_edges):
    """
    Generates human-readable labels for each bucket based on bucket edges.

    Parameters:
    - bucket_edges (list): The precomputed bucket edges.

    Returns:
    - list: A list of bucket labels in the form of intervals.
    """
    bucket_labels = []
    for i in range(len(bucket_edges) + 1):
        if i == 0:
            label = f"< {bucket_edges[i]:.0f} words"
        elif i == len(bucket_edges):
            label = f">= {bucket_edges[i-1]:.0f} words"
        else:
            label = f"{bucket_edges[i-1]:.0f} - {bucket_edges[i]:.0f} words"
        bucket_labels.append(label)

    return bucket_labels


def bucketize_word_count(text, bucket_edges):
    """
    Assigns a human-readable bucket label to a given text based on precomputed bucket edges.

    Parameters:
    - text (str): The input text.
    - bucket_edges (list): The precomputed bucket edges.

    Returns:
    - str: The bucket label corresponding to the word count.
    """
    word_count = len(text.split())
    bucket_index = np.digitize(word_count, bucket_edges)

    # Generate bucket labels
    bucket_labels = label_buckets(bucket_edges)

    return bucket_labels[bucket_index]


def sort_buckets(df, bucket_column, bucket_edges):
    """
    Sorts a DataFrame by the bucket intervals in the correct order.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the bucket column.
    - bucket_column (str): The name of the column containing bucket labels.
    - bucket_edges (list): The precomputed bucket edges.

    Returns:
    - pd.DataFrame: The sorted DataFrame.
    """
    # Generate ordered bucket labels
    ordered_labels = label_buckets(bucket_edges)

    # Convert the bucket column to a categorical type with a defined order
    df[bucket_column] = pd.Categorical(
        df[bucket_column], categories=ordered_labels, ordered=True
    )

    # Sort the DataFrame
    df_sorted = df.sort_values(by=bucket_column)

    return df_sorted
