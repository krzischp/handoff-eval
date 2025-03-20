import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Investigation plots -----


def compute_model_metrics_df(matched_pairs_dict, metric=None, error_type="mae"):
    """
    Computes precision or error metrics (MAE, MAPE, R², Balanced Recall-R²) for different models and examples.

    Returns:
    - A DataFrame containing columns: ["example", "model", "metric", "error_type", "value"].
    """
    valid_metrics = ["qty", "rateUsd", "rowTotalCostUsd"]
    valid_error_types = ["mae", "mape", "r2", "recall", "precision"]

    if metric not in valid_metrics:
        metric = None  # Default to None, meaning we're dealing with recall/precision

    if error_type not in valid_error_types:
        error_type = "mae"  # Default to MAE if invalid

    data = []
    for model_name, examples in matched_pairs_dict.items():
        for example_name, example_data in examples.items():
            df = example_data["matched_pairs_data"]

            # Handle recall and precision separately
            if error_type in ["recall", "precision"]:
                value = example_data.get(error_type, None)  # Fetch recall or precision
            else:
                # Compute the chosen error metric
                if metric is not None:
                    gt_col = f"gt_{metric}"
                    pred_col = f"pred_{metric}"

                    if gt_col in df.columns and pred_col in df.columns:
                        gt_values = df[gt_col].values
                        pred_values = df[pred_col].values

                        if error_type == "mae":
                            value = np.mean(np.abs(gt_values - pred_values))
                        elif error_type == "mape":
                            gt_values = np.where(
                                gt_values == 0, 1e-6, gt_values
                            )  # Avoid division by zero
                            value = (
                                np.mean(np.abs((gt_values - pred_values) / gt_values))
                                * 100
                            )  # In percentage
                        elif error_type == "r2":
                            value = r2_score(gt_values, pred_values)
                    else:
                        value = None  # Handle missing columns
                else:
                    value = None  # No valid metric provided

            data.append(
                {
                    "example": example_name,
                    "model": model_name,
                    "metric": (
                        metric if metric else error_type
                    ),  # If metric is None, store recall/precision
                    "error_type": error_type,
                    "value": value,
                }
            )

    return pd.DataFrame(data)


def aggregate_by_model(df):
    """
    Aggregates metric values by model, computing the mean across all examples.

    Parameters:
    - df: DataFrame containing ["example", "model", "value"].

    Returns:
    - plot_data: Aggregated DataFrame with mean values per model.
    - model_order: Sorted list of model names.
    """
    plot_data = df.groupby("model", as_index=False)[
        "value"
    ].mean()  # Aggregate over examples
    model_order = sorted(
        plot_data["model"].unique(), key=lambda x: int(x)
    )  # Order models numerically

    return plot_data, model_order


def filter_metrics(df_metrics, example_list=None, model_list=None):
    """
    Filters the df_metrics dataframe based on a list of examples and/or models.

    Parameters:
    - df_metrics (pd.DataFrame): The dataframe containing evaluation metrics.
    - example_list (list, optional): List of examples to keep (e.g., ["example_01", "example_02"]).
    - model_list (list, optional): List of models to keep (e.g., [2, 4, 5]).

    Returns:
    - pd.DataFrame: The filtered dataframe.
    """
    filtered_df = df_metrics.copy()

    if example_list is not None:
        filtered_df = filtered_df[filtered_df["example"].isin(example_list)]

    if model_list is not None:
        filtered_df = filtered_df[filtered_df["model"].isin(model_list)]

    return filtered_df


# Example usage:
# df_metrics = add_metadata_to_metrics(df_metrics, ground_truth_data, lambda x: bucketize_word_count(x, 3), "word_count_bucket")
# display(df_metrics) to visualize the updated dataframe


def add_metadata_to_metrics(df_metrics, ground_truth_data, metadata_func, column_name):
    """
    Adds a new metadata column to df_metrics based on a function applied to ground_truth_data.

    Parameters:
    - df_metrics (pd.DataFrame): The dataframe containing evaluation metrics.
    - ground_truth_data (dict): Dictionary containing ground truth data with "input" texts.
    - metadata_func (function): Function to extract metadata from the input text.
    - column_name (str): The name of the new column to be added to df_metrics.

    Returns:
    - pd.DataFrame: The updated dataframe with the new metadata column.
    """
    df_metrics_copy = df_metrics.copy()

    # Compute metadata for each example in ground_truth_data
    metadata_dict = {
        example: metadata_func(data["input"])
        for example, data in ground_truth_data.items()
    }

    # Map metadata values to the copied dataframe
    df_metrics_copy[column_name] = df_metrics_copy["example"].map(metadata_dict)

    return df_metrics_copy


def plot_model_metrics(df, x="example", metric_name="Metric Value"):
    """
    Plots a DataFrame containing model evaluation metrics.

    Parameters:
    - df: DataFrame containing ["example", "model", "value"].
    - x: "example" to plot per example, or "model" to aggregate over all examples.
    """
    valid_x_types = ["example", "model"]
    if x not in valid_x_types:
        x = "example"  # Default to example-based plotting

    # Sort x-axis and hue before plotting
    if x == "example":
        x_label = x.capitalize()
        example_order = sorted(df["example"].unique())  # Order examples alphabetically
        model_order = sorted(
            df["model"].unique(), key=lambda x: int(x)
        )  # Order models numerically
        plot_data = df
    else:  # Aggregate by model
        x_label = "Model"
        plot_data, model_order = aggregate_by_model(df)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_data,
        x=x,
        y="value",
        hue=(
            "model" if x == "example" else None
        ),  # Use hue for models only if plotting by example
        order=example_order if x == "example" else model_order,
    )

    # Formatting
    plt.xlabel(x_label)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} by {x_label}")
    plt.xticks(rotation=45)

    if x == "example":
        plt.legend(title="Model")

    # Show plot
    plt.show()


def plot_metric_by_metadata(df_metrics, error_type, x_column, hue_column=None):
    """
    Plots a selected metric from df_metrics with specified x-axis and optional hue.

    Parameters:
    - df_metrics (pd.DataFrame): The dataframe containing evaluation metrics.
    - metric (str): The metric to plot (e.g., "recall", "mape").
    - x_column (str): The column to use for the x-axis (e.g., "example", "model").
    - hue_column (str, optional): The column to use for hue (e.g., "model").

    Displays:
    - A sorted bar plot of the selected metric.
    """

    # Filter for the selected metric
    df_filtered = df_metrics[df_metrics["error_type"] == error_type].copy()

    # Ensure sorting of x-axis and hue
    df_filtered.sort_values(
        by=[x_column, hue_column] if hue_column else [x_column], inplace=True
    )

    # Plot the metric
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_filtered, x=x_column, y="value", hue=hue_column, errorbar=("ci", 95)
    )
    plt.xticks(rotation=45)
    plt.xlabel(x_column)
    plt.ylabel(error_type)
    plt.title(f"{error_type} by {x_column}")
    plt.legend(title=hue_column)
    plt.show()


# ---- Find best tradeoff ----
def find_best_tradeoff_models(matched_pairs_dict, top_n=2):
    """
    Finds the best tradeoff models based on highest recall and lowest MAPE.

    Parameters:
    - matched_pairs_dict (dict): Dictionary containing model evaluation data.
    - top_n (int): Number of top models to select.

    Returns:
    - list: Intersection of top models based on highest recall and lowest MAPE.
    """

    # Compute recall metrics
    error_type_recall = "recall"
    df_metrics_recall = compute_model_metrics_df(
        matched_pairs_dict, metric=None, error_type=error_type_recall
    )
    agg_recall, _ = aggregate_by_model(df_metrics_recall)
    top_recall_models = set(agg_recall.nlargest(top_n, "value")["model"])

    # Compute MAPE metrics
    metric_mape = "rowTotalCostUsd"
    error_type_mape = "mape"
    df_metrics_mape = compute_model_metrics_df(
        matched_pairs_dict, metric=metric_mape, error_type=error_type_mape
    )
    agg_mape, _ = aggregate_by_model(df_metrics_mape)
    best_mape_models = set(agg_mape.nsmallest(top_n, "value")["model"])

    # Find intersection of best models
    best_tradeoff_models = top_recall_models.intersection(best_mape_models)

    return best_tradeoff_models
