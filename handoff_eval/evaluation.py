import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

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


def plot_model_metrics(df, x="example", metric_name="Metric Value"):
    import matplotlib.pyplot as plt

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
