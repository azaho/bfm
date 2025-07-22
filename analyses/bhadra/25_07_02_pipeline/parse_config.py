import json
import sys
import evaluation.neuroprobe.config as neuroprobe_config

def sh_array(name, values):
    quoted = ' '.join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)
    return f'{name}=({quoted})'

# Load model config JSON
config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)

# Model info
print(f'PRIMARY_TITLE="{config["primary_model"]["title"]}"')
print(f'PRIMARY_PATH="{config["primary_model"]["path"]}"')
print(f'PRIMARY_EVAL_RESULTS_PATH="{config["primary_model"]["eval_results_path"]}"')
print(f'{sh_array("PRIMARY_EPOCHS", config["primary_model"]["epochs"])}')

# Tasks
if "tasks" in config:
    tasks = config["tasks"]
elif hasattr(neuroprobe_config, "NEUROPROBE_TASKS"):
    tasks = neuroprobe_config.NEUROPROBE_TASKS
else:
    raise ValueError("Tasks not found in config and not available in neuroprobe_config.")
print(f'{sh_array("TASKS", tasks)}')

# Use BTBENCH_LITE_SUBJECT_TRIALS from neuroprobe_config
subject_trials = neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
subjects, trials = zip(*subject_trials)
print(f'{sh_array("subjects", subjects)}')
print(f'{sh_array("trials", trials)}')

# Output dirs
print(f'EVAL_RESULTS_ROOT="{config["eval_results_output_root"]}"')
print(f'FIGURE_OUTPUT_DIR="{config["figure_output_dir"]}"')

# Comparison models — still from JSON config
print(f'COMPARISON_MODELS=\'{json.dumps(config["comparison_models"])}\'')

# Pass full subject_trials list to Python scripts
print(f'SUBJECT_TRIALS=\'{json.dumps(subject_trials)}\'')