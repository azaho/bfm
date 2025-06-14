import torch
import os, json
import numpy as np
from model.andrii_original_model import OriginalModel
from model.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from dataset import load_dataloaders, load_subjects
from evaluation.neuroprobe_tasks import FrozenModelEvaluation_SS_SM
from utils.training_config import log, convert_dtypes, unconvert_dtypes

def load_model_and_config(model_dir, epoch=None):
    """Load model and config from a saved directory."""
    # Load the checkpoint
    if epoch is None: epoch = "final"
    checkpoint_path = os.path.join("runs/data", model_dir, f"model_epoch_{epoch}.pth")
    checkpoint = torch.load(checkpoint_path)
    
    # Load config
    config = unconvert_dtypes(checkpoint['config'])
    
    # Initialize model and electrode embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OriginalModel(
        d_model=config['model']['transformer']['d_model'],
        n_layers_electrode=config['model']['transformer']['n_layers_electrode'],
        n_layers_time=config['model']['transformer']['n_layers_time'],
        n_heads=config['model']['transformer']['n_heads'],
        dropout=config['training']['dropout']
    ).to(device, dtype=config['model']['dtype'])
    
    electrode_embeddings = {
        'learned': ElectrodeEmbedding_Learned,
        'zero': ElectrodeEmbedding_Zero,
        'coordinate_init': ElectrodeEmbedding_Learned_CoordinateInit,
        'noisy_coordinate': ElectrodeEmbedding_NoisyCoordinate,
    }[config['model']['electrode_embedding']['type']](
        config['model']['transformer']['d_model'], 
        embedding_dim=config['model']['electrode_embedding']['dim'],
        coordinate_noise_std=config['model']['electrode_embedding']['coordinate_noise_std'],
    ).to(device, dtype=config['model']['dtype'])
    
    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    electrode_embeddings.load_state_dict(checkpoint['electrode_embeddings_state_dict'])
    
    return model, electrode_embeddings, config

def generate_frozen_features(batch, model, electrode_embeddings, config):
    """Generate features for evaluation."""
    device = next(model.parameters()).device
    electrode_indices = []
    subject_identifier = batch['subject_identifier'][0]
    for electrode_label in batch['electrode_labels'][0]:
        key = (subject_identifier, electrode_label)
        electrode_indices.append(electrode_embeddings.embeddings_map[key])
    batch['electrode_index'] = torch.tensor(electrode_indices, device=device, dtype=torch.long).unsqueeze(0).expand(batch['data'].shape[0], -1)
        
    if config['model']['signal_preprocessing']['laplacian_rereference']:
        from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
        batch = laplacian_rereference_batch(batch, remove_non_laplacian=False)
    
    if config['model']['signal_preprocessing']['normalize_voltage']:
        batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
        batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)

    embeddings = electrode_embeddings(batch['electrode_index'])
    features = model(batch['data'], embeddings, evaluation_features=True)

    if config['cluster']['eval_aggregation_method'] == 'mean':
        features = features.mean(dim=[1, 2])
    elif config['cluster']['eval_aggregation_method'] == 'concat':
        features = features.reshape(batch['data'].shape[0], -1)
    return features

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model')
    args = parser.parse_args()
    
    # Load model and config
    model, electrode_embeddings, config = load_model_and_config(args.model_dir)
    device = next(model.parameters()).device
    
    # Load subjects
    log(f"Loading subjects...", priority=0)
    all_subjects = load_subjects([], # leave the train subjects empty, we're only evaluating on the test subjects
                               config['training']['eval_subject_trials'], 
                               config['training']['data_dtype'], 
                               cache=config['cluster']['cache_subjects'], 
                               allow_corrupted=False)
    
    # Add subjects to electrode embeddings
    for subject in all_subjects.values():
        log(f"Adding subject {subject.subject_identifier} to electrode embeddings...", priority=0)
        electrode_embeddings.add_subject(subject)
    electrode_embeddings = electrode_embeddings.to(device, dtype=config['model']['dtype'])
    
    # Set up evaluation tasks
    eval_tasks = ['gpt2_surprisal', 'speech']  # Add more tasks as needed
    
    # Create evaluation function
    def eval_fn(batch):
        return generate_frozen_features(batch, model, electrode_embeddings, config)
    
    # Initialize evaluator
    evaluation = FrozenModelEvaluation_SS_SM(
        model_evaluation_function=eval_fn,
        eval_names=eval_tasks,
        lite=True,
        subject_trials=[(all_subjects[subject_identifier], trial_id) 
                       for subject_identifier, trial_id in config['training']['eval_subject_trials']],
        device=device,
        dtype=config['training']['data_dtype'],
        batch_size=config['training']['batch_size'],
        num_workers_eval=config['cluster']['num_workers_eval'],
        prefetch_factor=config['cluster']['prefetch_factor'],
    )
    
    # Run evaluation
    log("Running evaluation...", priority=0)
    model.eval()
    electrode_embeddings.eval()
    with torch.no_grad():
        eval_results = evaluation.evaluate_on_all_metrics(quick_eval=config['cluster']['quick_eval'])
        log(f"Evaluation results: {eval_results}", priority=0)
    
    # Save results
    results_path = os.path.join(args.model_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    log(f"Results saved to {results_path}", priority=0)

if __name__ == "__main__":
    main()
