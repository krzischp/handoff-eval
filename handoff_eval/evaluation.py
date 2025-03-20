import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

from handoff_eval.data_preparation import filter_similar_task

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


def aggregate_by_model(df, confidence=None):
    """
    Aggregates metric values by model, computing the mean and confidence interval across all examples.

    Parameters:
    - df: DataFrame containing ["example", "model", "value"].
    - confidence: Confidence level for the interval. If None, no confidence interval is computed.

    Returns:
    - plot_data: Aggregated DataFrame with mean values and confidence intervals per model (if applicable).
    - model_order: Sorted list of model names.
    """
    grouped = df.groupby("model")["value"]
    means = grouped.mean()

    plot_data = pd.DataFrame({"model": means.index, "mean_value": means.values})

    if confidence is not None:
        stds = grouped.std()
        counts = grouped.count()
        # - stats.t.ppf Student's t-distribution percent point function (inverse of CDF)
        # - (1 + confidence) / 2.0 Example: If confidence=0.95, then (1 + 0.95) / 2 = 0.975, \
        # which corresponds to the 97.5th percentile of the t-distribution
        # - counts - 1 is the degrees of freedom (n-1), where counts represents \
        # the number of data points for each model
        ci_half_width = stats.t.ppf((1 + confidence) / 2.0, counts - 1) * (
            stds / np.sqrt(counts)
        )

        plot_data["ci_lower"] = (means - ci_half_width).values
        plot_data["ci_upper"] = (means + ci_half_width).values

    model_order = sorted(
        plot_data["model"].unique(), key=lambda x: int(x)
    )  # Order models numerically

    return plot_data, model_order


def plot_model_metrics(df, x="example", metric_name="Metric Value", confidence=None):
    """
    Plots a DataFrame containing model evaluation metrics with optional confidence intervals.

    Parameters:
    - df: DataFrame containing ["example", "model", "value"].
    - x: "example" to plot per example, or "model" to aggregate over all examples.
    - confidence: Confidence level for the interval. If None, confidence intervals are not plotted.
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
        plot_data, model_order = aggregate_by_model(df, confidence=confidence)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_data,
        x=x if x == "example" else "model",
        y="value" if x == "example" else "mean_value",
        hue="model" if x == "example" else None,
        order=example_order if x == "example" else model_order,
    )

    # Add error bars manually if confidence intervals are calculated
    if x == "model" and confidence is not None:
        for i, row in plot_data.iterrows():
            ax.errorbar(
                x=i,
                y=row["mean_value"],
                yerr=[
                    [row["mean_value"] - row.get("ci_lower", row["mean_value"])],
                    [row.get("ci_upper", row["mean_value"]) - row["mean_value"]],
                ],
                fmt="none",
                ecolor="black",
                capsize=5,
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
def find_best_tradeoff_models(matched_pairs_dict, top_n=2, confidence=None):
    """
    Finds the best tradeoff models based on highest recall and lowest MAPE.

    Parameters:
    - matched_pairs_dict (dict): Dictionary containing model evaluation data.
    - top_n (int): Number of top models to select.
    - confidence (float or None): Confidence level for computing confidence intervals.

    Returns:
    - list: Intersection of top models based on highest recall and lowest MAPE.
    """

    # Compute recall metrics
    error_type_recall = "recall"
    df_metrics_recall = compute_model_metrics_df(
        matched_pairs_dict, metric=None, error_type=error_type_recall
    )
    agg_recall, _ = aggregate_by_model(df_metrics_recall, confidence=confidence)
    top_recall_models = set(agg_recall.nlargest(top_n, "mean_value")["model"])

    # Compute MAPE metrics
    filtered_matched_pairs_dict = filter_similar_task(matched_pairs_dict)
    metric_mape = "rowTotalCostUsd"
    error_type_mape = "mape"
    df_metrics_mape = compute_model_metrics_df(
        filtered_matched_pairs_dict, metric=metric_mape, error_type=error_type_mape
    )
    agg_mape, _ = aggregate_by_model(df_metrics_mape, confidence=confidence)
    best_mape_models = set(agg_mape.nsmallest(top_n, "mean_value")["model"])

    # Find intersection of best models
    best_tradeoff_models = top_recall_models.intersection(best_mape_models)

    return best_tradeoff_models


def plot_avg_estimation_time(model_output_data):
    """
    Computes and plots the average estimation time for each model.

    Parameters:
    - model_output_data (dict): Dictionary containing model outputs with `time_to_estimate_sec` values.
    """
    # Compute average estimation time for each model
    avg_times = {
        model_name: sum(
            pred["time_to_estimate_sec"] for pred in model_data["estimate_preds"]
        )
        / len(model_data["estimate_preds"])
        for model_name, model_data in model_output_data.items()
        if len(model_data["estimate_preds"]) > 0
    }

    # Convert to DataFrame and sort by time
    df_avg_time = pd.DataFrame(avg_times.items(), columns=["model", "avg_time"])
    df_avg_time = df_avg_time.sort_values(by="avg_time", ascending=True)

    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df_avg_time,
        x="model",
        y="avg_time",
        order=df_avg_time["model"].astype(str),
    )
    plt.xlabel("Model")
    plt.ylabel("Average Estimation Time (sec)")
    plt.title("Average Time to Estimate per Model")
    plt.xticks(rotation=45)
    plt.show()

    return df_avg_time
