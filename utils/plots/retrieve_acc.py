import os
import json
from collections import defaultdict

# Define the models and their respective pretrained versions
models_pretrained = {
    "ViT-H-14": "laion2B-s32B-b79K",
    "ViT-L-14": "laion2B-s32B-b82K",
    "ViT-B-16": "laion2B-s34B-b88K",
    "ViT-B-32": "laion2B-s34B-b79K"
}

# Define the max_text values to iterate over
max_text_values = [10, 20, 30, 40, 50, 60, 70, 80]

# Define the algorithm to run
algorithms = ["svd_data_approx", "text_span"]

# Base command template
base_command = (
    "python -u -m utils.scripts.compute_text_explanations "
    "--device cpu "
    "--seed 0 "
    "--num_of_last_layers 4 "
    "--text_descriptions top_1500_nouns_5_sentences_imagenet_clean "
)

# Iterate over each model and each max_text value
for algo in algorithms:
    for model, pretrained in models_pretrained.items():
        for max_text in max_text_values:
            # Construct the command for the current model and max_text
            command = (
                f"{base_command} "
                f"--model {model} "
                f"--max_text {max_text} "
                f"--algorithm {algo} "
            )
            # Execute the command
            print(f"Running: {command}")
            os.system(command)


### NOW AGGREGATE THE RESULTS ###

# Define the output directory and input file template
output_dir = "output_dir"
input_file_template = (
    "imagenet_completeness_top_1500_nouns_5_sentences_imagenet_clean_{model}_algo_{algo}_seed_0.jsonl"
)

# Data structure to hold results
results = {
    "ViT-H-14": {"text_span":[], "svd_data_approx":[]},
    "ViT-L-14": {"text_span":[], "svd_data_approx":[]},
    "ViT-B-16": {"text_span":[], "svd_data_approx":[]},
    "ViT-B-32": {"text_span":[], "svd_data_approx":[]},
    "max_text_values": max_text_values,
}

# Process each model's file
for model, pretrained in models_pretrained.items():
    for algo in algorithms:
        input_file = os.path.join(output_dir, input_file_template.format(model=model,algo=algo))
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue

        with open(input_file, "r") as json_file:
            for line in json_file:
                entry = json.loads(line)
                if entry["head"] == -1:  # Extract only if head is -1
                    accuracy = entry["accuracy"]
                    results[model][algo].append(accuracy)

# Save results to a JSON file
output_file = os.path.join(output_dir, "aggregated_accuracies.json")
with open(output_file, "w") as outfile:
    json.dump(results, outfile, indent=4)

print(f"Aggregated accuracies saved to {output_file}")
