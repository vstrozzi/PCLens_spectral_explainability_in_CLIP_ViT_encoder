import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Path to the JSON file with accuracy data
input_file = "output_dir/aggregated_accuracies.json"

# Load data from the JSON file
with open(input_file, "r") as file:
    data = json.load(file)

# Extract data for visualization
models = ["ViT-H-14", "ViT-L-14", "ViT-B-16", "ViT-B-32"]
max_text_values = data["max_text_values"]

# Example vanilla accuracy (replace with actual values)
vanilla_accuracies = {
    "ViT-H-14": 0.92,
    "ViT-L-14": 0.90,
    "ViT-B-16": 0.88,
    "ViT-B-32": 0.85,
}

# Extract accuracies for algorithms
algorithm_1_accuracies = [data[model] for model in models]
algorithm_2_accuracies = [
    data["algorithm_accuracies"][model] for model in models
]

# Compute accuracy differences
accuracy_differences = np.array(algorithm_1_accuracies) - np.array(algorithm_2_accuracies)

# Append vanilla accuracies as a new column
accuracy_differences_with_vanilla = np.column_stack(
    [accuracy_differences, [vanilla_accuracies[model] for model in models]]
)

# Add 'Vanilla Accuracy' as a new column label
heatmap_columns = max_text_values + ["Vanilla Accuracy"]

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    accuracy_differences_with_vanilla,
    annot=True,
    fmt=".2f",
    xticklabels=heatmap_columns,
    yticklabels=models,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Accuracy Difference (Algorithm 1 - Algorithm 2)"},
)
plt.title("Accuracy Difference Between Algorithms and Vanilla Accuracy")
plt.xlabel("Max Text Values + Vanilla Accuracy")
plt.ylabel("Models")
plt.tight_layout()
plt.show()
