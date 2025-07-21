# Train vs Test Loss Curves for Sweep Analysis
# Copy this code into your plot_sweeps.ipynb notebook

import matplotlib.pyplot as plt
import numpy as np

# Plot train vs test loss curves for all sweep combinations
def plot_sweep_loss_curves(results, model_name, weight_decay_options, dropout_options):
    """
    Create a subplot grid showing train vs test loss curves for all (wd, dr) combinations.
    
    Args:
        results: Dictionary with (wd, dr, model) keys and stats values
        model_name: Name of the model to plot
        weight_decay_options: List of weight decay values
        dropout_options: List of dropout values
    """
    fig, axes = plt.subplots(len(weight_decay_options), len(dropout_options), 
                             figsize=(20, 16))
    fig.suptitle(f'Train vs Test Loss Curves for {model_name} Sweep', fontsize=16)

    # Handle single row/column case
    if len(weight_decay_options) == 1:
        axes = axes.reshape(1, -1)
    if len(dropout_options) == 1:
        axes = axes.reshape(-1, 1)

    for i, wd in enumerate(weight_decay_options):
        for j, dr in enumerate(dropout_options):
            ax = axes[i, j]
            stats = results.get((wd, dr, model_name))
            
            if stats:
                # Extract train and test losses
                train_losses = [entry['batch_loss'] for entry in stats if 'batch_loss' in entry]
                test_losses = [entry['test_loss'] for entry in stats if 'test_loss' in entry]
                
                if train_losses and test_losses:
                    # Reshape train losses if needed
                    n_epochs = len(test_losses)
                    if len(train_losses) >= n_epochs * 91:  # Assuming 91 batches per epoch
                        train_losses = np.reshape(train_losses[:(n_epochs*91)], (n_epochs, 91))
                        train_losses = train_losses.mean(axis=1)
                    else:
                        train_losses = train_losses[:n_epochs]
                    
                    # Plot
                    epochs = range(1, len(test_losses) + 1)
                    ax.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
                    ax.plot(epochs, test_losses, label='Test Loss', color='red', alpha=0.7)
                    
                    ax.set_title(f'wd={wd}, dr={dr}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'wd={wd}, dr={dr}')

    plt.tight_layout()
    plt.show()

# Usage example:
# plot_sweep_loss_curves(results, "andrii0", weight_decay_options, dropout_options)
# plot_sweep_loss_curves(results, "roshnipm_pair_nocommon", weight_decay_options, dropout_options)

# Alternative: Overlay all curves on one plot for easier comparison
def plot_overlayed_curves(results, model_name, weight_decay_options, dropout_options):
    """
    Plot all train and test loss curves on the same plot with different colors/linestyles.
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(weight_decay_options) * len(dropout_options)))
    color_idx = 0
    
    for wd in weight_decay_options:
        for dr in dropout_options:
            stats = results.get((wd, dr, model_name))
            
            if stats:
                train_losses = [entry['batch_loss'] for entry in stats if 'batch_loss' in entry]
                test_losses = [entry['test_loss'] for entry in stats if 'test_loss' in entry]
                
                if train_losses and test_losses:
                    n_epochs = len(test_losses)
                    if len(train_losses) >= n_epochs * 91:
                        train_losses = np.reshape(train_losses[:(n_epochs*91)], (n_epochs, 91))
                        train_losses = train_losses.mean(axis=1)
                    else:
                        train_losses = train_losses[:n_epochs]
                    
                    epochs = range(1, len(test_losses) + 1)
                    plt.plot(epochs, train_losses, '--', color=colors[color_idx], 
                            alpha=0.7, label=f'Train (wd={wd}, dr={dr})')
                    plt.plot(epochs, test_losses, '-', color=colors[color_idx], 
                            alpha=0.7, label=f'Test (wd={wd}, dr={dr})')
                    color_idx += 1
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'All Loss Curves for {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_overlayed_curves(results, "andrii0", weight_decay_options, dropout_options) 

def plot_sweep_eval_curves(results, model_name, weight_decay_options, dropout_options, eval_keys=None):
    """
    Create a subplot grid showing eval AUROC curves for all (wd, dr) combinations.
    Args:
        results: Dictionary with (wd, dr, model) keys and stats values
        model_name: Name of the model to plot
        weight_decay_options: List of weight decay values
        dropout_options: List of dropout values
        eval_keys: List of eval keys to plot (default: ['eval_auroc/average_onset', 'eval_auroc/average_gpt2_surprisal', 'eval_auroc/average_overall'])
    """
    if eval_keys is None:
        eval_keys = ['eval_auroc/average_onset', 'eval_auroc/average_gpt2_surprisal', 'eval_auroc/average_overall']
    eval_labels = {
        'eval_auroc/average_onset': 'Onset',
        'eval_auroc/average_gpt2_surprisal': 'Surprisal',
        'eval_auroc/average_overall': 'Overall',
    }
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(len(weight_decay_options), len(dropout_options), figsize=(20, 16))
    fig.suptitle(f'Eval AUROC Curves for {model_name} Sweep', fontsize=16)

    if len(weight_decay_options) == 1:
        axes = axes.reshape(1, -1)
    if len(dropout_options) == 1:
        axes = axes.reshape(-1, 1)

    for i, wd in enumerate(weight_decay_options):
        for j, dr in enumerate(dropout_options):
            ax = axes[i, j]
            stats = results.get((wd, dr, model_name))
            if stats:
                epochs = [entry['epoch'] for entry in stats if 'epoch' in entry and all(k in entry for k in eval_keys)]
                for k, key in enumerate(eval_keys):
                    values = [entry[key] for entry in stats if 'epoch' in entry and key in entry]
                    if values:
                        # Try to infer x-axis (epochs) for evals (often every 5 epochs)
                        if len(values) == len(epochs):
                            ax.plot(epochs, values, label=eval_labels.get(key, key), color=colors[k], alpha=0.8)
                        else:
                            ax.plot(np.arange(1, len(values)+1), values, label=eval_labels.get(key, key), color=colors[k], alpha=0.8)
                ax.set_title(f'wd={wd}, dr={dr}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('AUROC')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'wd={wd}, dr={dr}')
    plt.tight_layout()
    plt.show()

def plot_overlayed_eval_curves(results, model_name, weight_decay_options, dropout_options, eval_key='eval_auroc/average_overall'):
    """
    Overlay all eval AUROC curves for a given eval_key on one plot for all sweep combinations.
    Args:
        results: Dictionary with (wd, dr, model) keys and stats values
        model_name: Name of the model to plot
        weight_decay_options: List of weight decay values
        dropout_options: List of dropout values
        eval_key: Which eval key to plot (default: 'eval_auroc/average_overall')
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(weight_decay_options) * len(dropout_options)))
    color_idx = 0
    for wd in weight_decay_options:
        for dr in dropout_options:
            stats = results.get((wd, dr, model_name))
            if stats:
                values = [entry[eval_key] for entry in stats if eval_key in entry]
                epochs = [entry['epoch'] for entry in stats if eval_key in entry and 'epoch' in entry]
                if values:
                    if len(values) == len(epochs):
                        plt.plot(epochs, values, '-', color=colors[color_idx], alpha=0.8, label=f'wd={wd}, dr={dr}')
                    else:
                        plt.plot(np.arange(1, len(values)+1), values, '-', color=colors[color_idx], alpha=0.8, label=f'wd={wd}, dr={dr}')
                    color_idx += 1
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title(f'Overlayed {eval_key} Curves for {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_sweep_eval_curves(results, "andrii0", weight_decay_options, dropout_options)
# plot_overlayed_eval_curves(results, "andrii0", weight_decay_options, dropout_options, eval_key='eval_auroc/average_overall') 