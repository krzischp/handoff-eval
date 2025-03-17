import pandas as pd
import seaborn as sns
import numpy as np

# ----- Investigation plots -----


def compute_model_metrics_df(matched_pairs_dict, metric=None, error_type="mae"):
    from sklearn.metrics import r2_score

    """
    Computes recall or error metrics (MAE, MAPE, R², Balanced Recall-R²) for different models and examples.

    Returns:
    - A DataFrame containing columns: ["example", "model", "value"].
    """
    valid_metrics = ["qty", "rateUsd", "rowTotalCostUsd"]
    valid_error_types = ["mae", "mape", "r2", "balanced"]

    if metric not in valid_metrics:
        metric = "recall"  # Default to recall if metric is None or invalid

    if error_type not in valid_error_types:
        error_type = "mae"  # Default to MAE if error_type is invalid

    data = []
    for model_name, examples in matched_pairs_dict.items():
        for example_name, example_data in examples.items():
            df = example_data["matched_pairs_data"]
            recall = example_data["recall"]

            if metric == "recall":
                value = recall
            else:
                # Compute the chosen error metric
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
                            np.mean(np.abs((gt_values - pred_values) / gt_values)) * 100
                        )  # In percentage
                    elif error_type == "r2":
                        value = r2_score(gt_values, pred_values)
                    elif error_type == "balanced":
                        r2_value = r2_score(gt_values, pred_values)
                        r2_normalized = (
                            r2_value + 1
                        ) / 2  # Normalize R² to (0,1) range
                        value = (
                            np.sqrt(recall * r2_normalized)
                            if recall is not None and r2_value is not None
                            else None
                        )
                else:
                    value = None  # Handle missing columns

            data.append({"example": example_name, "model": model_name, "value": value})

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


def plot_model_metrics(df, x="example"):
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
        x_label = "Example"
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
    plt.ylabel("Metric Value")
    plt.title(f"Model Metrics by {x_label}")
    plt.xticks(rotation=45)

    if x == "example":
        plt.legend(title="Model")

    # Show plot
    plt.show()
