import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation import match_line_items


# Function to calculate similarity data for a single model
def calculate_similarity_for_model(model_name, model_data, ground_truth_data):
    similarity_data = []

    for prediction in model_data["estimate_preds"]:
        file_name = prediction["valid_file_name"]
        if file_name not in ground_truth_data:
            continue

        gt_rows = ground_truth_data[file_name]["rows"]
        pred_rows = prediction["rows"]

        matched_pairs = match_line_items(gt_rows, pred_rows)
        similarity_ratio = len(matched_pairs) / len(gt_rows) if gt_rows else 0

        similarity_data.append(
            {
                "Example": file_name,
                "Similarity Ratio": similarity_ratio,
                "Model": model_name,
            }
        )

    return similarity_data


# Function to calculate similarity data
def calculate_similarity_data(model_output_data, ground_truth_data):
    similarity_data = []

    for model_name, model_data in model_output_data.items():
        similarity_data.extend(
            calculate_similarity_for_model(model_name, model_data, ground_truth_data)
        )

    df = pd.DataFrame(similarity_data)
    df.sort_values(by=["Example", "Model"], ascending=True, inplace=True)
    return df


# Function to plot similarity data
def plot_similarity_ratios(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Example", y="Similarity Ratio", hue="Model", data=df)
    plt.xticks(rotation=45)
    plt.title("Similarity Ratios of Matched Line Items per Model")
    plt.xlabel("Example")
    plt.ylabel("Similarity Ratio")
    plt.legend(title="Model")
    plt.show()
