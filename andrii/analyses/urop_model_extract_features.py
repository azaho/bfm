import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to path

from subject_braintreebank import BrainTreebankSubject
import btbench.btbench_datasets as btbench_datasets
import btbench.btbench_config as btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--eval_names', type=str, default='onset', help='Evaluation names list, separated by commas (e.g. onset, gpt2_surprisal)')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='analyses/eval_features', help='Directory to save results')
parser.add_argument('--model_dir', type=str, help='Directory containing model files')
parser.add_argument('--model_epoch', type=int, default=100, help='Model epoch to evaluate')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
args = parser.parse_args()


eval_names = args.eval_names.split(',')
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
model_dir = args.model_dir
model_epoch = args.model_epoch
batch_size = args.batch_size
overwrite = bool(args.overwrite)

# Loading the model
import torch
from model_model import TransformerModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeDataEmbeddingFFT
from utils_train import *
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(f"models_data/{model_dir}/model_epoch_{model_epoch}.pth", map_location=device)

training_config, model_config, cluster_config = checkpoint['training_config'], checkpoint['model_config'], checkpoint['cluster_config']
training_config = unconvert_dtypes(training_config) # convert string dtypes back to torch dtypes
model_config = unconvert_dtypes(model_config)
cluster_config = unconvert_dtypes(cluster_config)

bins_start_before_word_onset_seconds = 0.5
bins_end_after_word_onset_seconds = 2.5
bin_size_seconds = model_config['sample_timebin_size']

# Initialize model
model = TransformerModel(
    model_config['transformer']['d_model'],
    n_layers_electrode=model_config['transformer']['n_layers_electrode'],
    n_layers_time=model_config['transformer']['n_layers_time'],
    use_cls_token=model_config['transformer']['use_cls_token']
).to(device, dtype=model_config['dtype'])

# Initialize electrode embeddings based on config type
if model_config['electrode_embedding']['type'] == 'learned':
    electrode_embeddings = ElectrodeEmbedding_Learned(
        model_config['transformer']['d_model'],
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
elif model_config['electrode_embedding']['type'] == 'coordinate_init':
    electrode_embeddings = ElectrodeEmbedding_Learned_CoordinateInit(
        model_config['transformer']['d_model'],
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
elif model_config['electrode_embedding']['type'] == 'noisy_coordinate':
    electrode_embeddings = ElectrodeEmbedding_NoisyCoordinate(
        model_config['transformer']['d_model'],
        coordinate_noise_std=model_config['electrode_embedding']['coordinate_noise_std'],
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
else:
    raise ValueError(f"Invalid electrode embedding type: {model_config['electrode_embedding']['type']}")

electrode_embeddings = electrode_embeddings.to(device, dtype=model_config['dtype'])

electrode_data_embeddings = ElectrodeDataEmbeddingFFT(
    electrode_embeddings,
    model_config['sample_timebin_size']
).to(device, dtype=model_config['dtype'])

# Load saved model checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
electrode_data_embeddings.load_state_dict(checkpoint['electrode_data_embeddings_state_dict'])
if verbose: print("Model loaded", model.eval(), "", sep="\n")


# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
all_electrode_labels = subject.electrode_labels

subject.load_neural_data(trial_id)
if verbose:
    print(f"Subject {subject.subject_identifier} trial {trial_id} loaded")

assert bin_size_seconds == model_config['sample_timebin_size'], f"Bin size {bin_size_seconds} does not match model config sample timebin size {model_config['sample_timebin_size']}"
assert int((bins_start_before_word_onset_seconds+bins_end_after_word_onset_seconds)/bin_size_seconds) <= model_config['max_n_timebins'], f"Time window {bins_start_before_word_onset_seconds+bins_end_after_word_onset_seconds} does is too big given the model config max_n_timebins {model_config['max_n_timebins']}"


for eval_name in eval_names:
    save_file_path = os.path.join(save_dir, model_dir+f"_epoch{model_epoch}", f"frozen_population_btbank{subject_id}_{trial_id}_{eval_name}.npy")
    if not overwrite and os.path.exists(save_file_path):
        if verbose:
            print(f"Skipping {save_file_path} because it already exists")
        continue
    
    dataset = btbench_datasets.BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=eval_name,
                                                                        output_indices=False, 
                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE))

    X_all_bins = []
    y = []

    # Pass through model to get all train and test outputs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_input, batch_label in dataloader:
            batch_input = batch_input.to(device, dtype=model_config['dtype'])
            electrode_data = electrode_data_embeddings.forward(subject.subject_identifier, subject.get_electrode_indices(), batch_input)
            model_output = model(electrode_data, only_electrode_output=True)[0]
            X_all_bins.append(model_output)
            y.append(batch_label)
   
    X_all_bins = torch.cat(X_all_bins, dim=0).unsqueeze(2).float().cpu().numpy() # shape: (n_dataset, n_timebins, 1, d_model)
    y = torch.cat(y, dim=0).float().cpu().numpy()

    # Save results as npy file
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    np.save(save_file_path, {
        'X': X_all_bins,
        'y': y,

        'metadata': {
            'subject_id': subject.subject_identifier,
            'trial_id': trial_id,
            'eval_name': eval_name,
            'training_config': checkpoint['training_config'],
            'model_config': checkpoint['model_config'],
            'cluster_config': checkpoint['cluster_config']
        }
    })
    if verbose:
        print(f"Saved results to {save_file_path}")
