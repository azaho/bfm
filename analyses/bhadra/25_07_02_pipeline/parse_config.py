import json
import sys

config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)

def sh_array(name, values):
    quoted = ' '.join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)
    return f'{name}=({quoted})'

print(f'PRIMARY_TITLE="{config["primary_model"]["title"]}"')
print(f'PRIMARY_PATH="{config["primary_model"]["path"]}"')
print(f'PRIMARY_EVAL_RESULTS_PATH="{config["primary_model"]["eval_results_path"]}"')
print(f'{sh_array("PRIMARY_EPOCHS", config["primary_model"]["epochs"])}')
print(f'{sh_array("TASKS", config["tasks"])}')

subjects, trials = zip(*config["subject_trials"])
print(f'{sh_array("subjects", subjects)}')
print(f'{sh_array("trials", trials)}')

print(f'EVAL_RESULTS_ROOT="{config["eval_results_output_root"]}"')
print(f'FIGURE_OUTPUT_DIR="{config["figure_output_dir"]}"')

# For comparison_models, emit one big JSON string
print(f'COMPARISON_MODELS=\'{json.dumps(config["comparison_models"])}\'')
print(f'SUBJECT_TRIALS=\'{json.dumps(config["subject_trials"])}\'')
