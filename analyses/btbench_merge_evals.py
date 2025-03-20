import os
import json
import glob
import time

# Get all relevant JSON files
eval_results_dir = "analyses/eval_results"
electrode_files = glob.glob(os.path.join(eval_results_dir, "frozen_transformer_model_electrode*.json"))
population_files = glob.glob(os.path.join(eval_results_dir, "frozen_transformer_model_population*.json"))

# Initialize merged results dictionary
merged_results = {
    "model_name": "Andrii's model",
    "author": "Andrii Zahorodnii", 
    "description": "Linear regression on top of the output of the transformer model.",
    "organization": "MIT",
    "organization_url": "https://azaho.org/",
    "timestamp": time.time(),
    "evaluation_results": {}
}

# Helper function to merge results
def merge_results(merged_results, new_results):
    for subject_trial, eval_data in new_results["evaluation_results"].items():
        if subject_trial not in merged_results["evaluation_results"]:
            merged_results["evaluation_results"][subject_trial] = {}
        merged_results["evaluation_results"][subject_trial].update(eval_data)

# Merge electrode files
for file_path in electrode_files:
    with open(file_path, 'r') as f:
        results = json.load(f)
        merge_results(merged_results, results)

# Merge population files  
for file_path in population_files:
    with open(file_path, 'r') as f:
        results = json.load(f)
        merge_results(merged_results, results)

# Save merged results
with open('analyses/eval_results/frozen_transformer_model.json', 'w') as f:
    json.dump(merged_results, f, indent=4)

# Also save a gzipped version with maximum compression
import gzip
with gzip.open('analyses/eval_results/frozen_transformer_model.json.gz', 'wt', compresslevel=9) as f:
    json.dump(merged_results, f, indent=4)
