import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import os
import pandas as pd

def plot_given_electrode_data(electrode_data, electrode_labels, times, frequencies=None, n_columns=5, annotations={}, plot_width=1, plot_height=1, save_path=None):
    # electrode_data: (n_electrodes, n_samples) OR (n_electrodes, n_timebins, n_frequency_bins)
    # electrode_labels: list of electrode labels
    # times: time vector (n_samples) OR (n_timebins)
    # n_columns: number of columns in the plot
    # plot_width: width of the plot (default 1)
    # plot_height: height of the plot (default 1)
    # save_path: path to save the plot
    assert electrode_data.shape[0] == len(electrode_labels), f"Number of electrodes in electrode_data ({electrode_data.shape[0]}) does not match number of electrodes in electrode_labels ({len(electrode_labels)})"
    assert electrode_data.shape[1] == len(times), f"Number of timebins in electrode_data ({electrode_data.shape[1]}) does not match number of timebins in times ({len(times)})"
    if electrode_data.ndim == 3:
        assert frequencies is not None, "Frequencies must be provided if electrode_data is 3D"
        assert electrode_data.shape[2] == len(frequencies), f"Number of frequency bins in electrode_data ({electrode_data.shape[2]}) does not match number of frequency bins in frequencies ({len(frequencies)})"
    else:
        assert frequencies is None, "Frequencies must be None if electrode_data is 2D"

    fm.fontManager.addfont('analyses/font_arial.ttf')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 12})
    
    n_electrodes = len(electrode_labels)
    n_rows = int(np.ceil(n_electrodes / n_columns))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(6*n_columns*plot_width, 1.4*n_rows*plot_height), sharex=True, sharey=True)
    axes = axes.flatten()

    if frequencies is not None:
        assert electrode_data.ndim == 3, f"Electrode data must be 3D if frequencies are provided"
        assert electrode_data.shape[2] == len(frequencies), f"Number of frequency bins in electrode_data ({electrode_data.shape[2]}) does not match number of frequency bins in frequencies ({len(frequencies)})"
    # Plot each electrode
    for i, ax in enumerate(axes):
        if i < n_electrodes:
            if electrode_data.ndim == 2:
                ax.axhline(y=0, color='gray', linestyle='--', alpha=1, linewidth=1)
                ax.plot(times, electrode_data[i], 'k-', linewidth=0.8)

                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                # ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Get global y limits
                y_scale = max(abs(electrode_data.min()), abs(electrode_data.max()))
                y_min, y_max = -y_scale, y_scale
                ax.set_ylim(y_min, y_max)

                ax.set_xlim(times[0], times[-1])

                if i % n_columns == 0:  # Only for first column
                    ax.set_ylabel('µV')
            else:
                ax.imshow(electrode_data[i].T, aspect='auto', origin='lower', cmap='viridis', 
                         interpolation='none', extent=[times[0], times[-1], frequencies[0], frequencies[-1]])

                ax.set_yticks(frequencies[torch.linspace(0, frequencies.shape[0]-1, 4).round().int()])
                if i % n_columns == 0:  # Only for first column
                    ax.set_ylabel('Freq (Hz)')
                    
            if times is not None:
                time_ticks = {
                    t.item(): f"{t:.1f}" for t in np.linspace(times[0], times[-1], 5)
                }
                time_ticks.update({
                    t: f"\n#{i+1}" if t not in time_ticks else f"{time_ticks[t]}\n#{i+1}" for i, t in enumerate(annotations.keys())
                })
                ax.set_xticks(list(time_ticks.keys()))
                ax.set_xticklabels(list(time_ticks.values()))
                
            ax.set_title(electrode_labels[i], fontsize=12)
            # Only show x label for bottom row if no annotations, otherwise second to last row
            if i >= len(axes) - n_columns:
                ax.set_xlabel('Time (s)')

            for annotation_index in annotations:
                ax.axvline(x=annotation_index, color='white' if electrode_data.ndim == 3 else 'gray', linestyle='--', alpha=1.0, linewidth=1)
        else:
            ax.axis('off')

    plt.tight_layout()
    if save_path is not None:
        if '.' not in save_path:
            save_path = save_path + '.pdf'
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_electrode_data(subject, trial_id, window_from, window_to, n_columns=3, annotations={}, electrodes=None, laplacian_rereference=True, spectrogram=False, spectrogram_parameters=None, spectrogram_normalization_parameters=None, time_start=None, time_end=None, plot_width=1, plot_height=1, save_path=None):
    """Plot neural data from multiple electrodes, either as time series or spectrograms.

    Args:
        subject: Subject object containing the neural data and metadata
        trial_id: ID of the session to take data from
        window_from: Start time of the window to plot (in samples)
        window_to: End time of the window to plot (in samples) 
        n_columns: Number of columns in the subplot grid (default: 3)
        annotations: Dictionary of time points to mark with vertical lines. Format: {time_seconds: annotation_text} (default: {})
        electrodes: List of electrode labels to plot. If None, plots all electrodes (default: None)
        laplacian_rereference: Whether to apply Laplacian rereferencing (default: True)
        spectrogram: Whether to plot spectrograms instead of time series (default: False)
        spectrogram_parameters: Parameters for spectrogram computation (default: None, which means the default parameters are used)
        spectrogram_normalization_parameters: Mean and std for spectrogram normalization. If None, the normalization is done within the plotted window.
        time_start: Start time for x-axis labels in seconds (default: None)
        time_end: End time for x-axis labels in seconds (default: None)
        plot_width: Width scaling factor for the plot (default: 1, which means the default width is used)
        plot_height: Height scaling factor for the plot (default: 1, which means the default height is used)
        save_path: Path to save the figure. If None, figure is not saved (default: None)
    """
    electrode_labels = subject.get_electrode_labels() if electrodes is None else electrodes
    electrode_ids = [subject.get_electrode_labels().index(label) for label in electrode_labels]
    electrode_locations = {electrode_label: subject.get_electrode_metadata(electrode_label)['DesikanKilliany'] for electrode_label in electrode_labels}
    electrode_titles = [f"{subject.subject_identifier}_{trial_id}_{label} ({location})" for label, location in electrode_locations.items()]
    
    electrode_data = subject.get_all_electrode_data(trial_id, window_from=window_from, window_to=window_to)
    electrode_data = electrode_data[electrode_ids] # reorder electrodes to match electrode_labels

    if laplacian_rereference:
        from model.preprocessing.laplacian_rereferencing import laplacian_rereference_neural_data
        electrode_data, _, _ = laplacian_rereference_neural_data(electrode_data, electrode_labels, remove_non_laplacian=False)

    if spectrogram:
        from model.preprocessing.spectrogram import SpectrogramPreprocessor
        spectrogram_preprocessor = SpectrogramPreprocessor(spectrogram_parameters=spectrogram_parameters)
        electrode_data, frequency_bins, time_bins = spectrogram_preprocessor({
            'data': electrode_data.unsqueeze(0), 
            'metadata': {
                'sampling_rate': subject.get_sampling_rate()
            }
        }, output_time_frequency_bins=True, z_score=spectrogram_normalization_parameters is None)
        times = time_bins
        electrode_data = electrode_data[0]

        if spectrogram_normalization_parameters is not None:
            mean, std = spectrogram_normalization_parameters
            mean = mean.unsqueeze(1) # n_electrodes_full, 1, n_frequency_bins
            std = std.unsqueeze(1) # n_electrodes_full, 1, n_frequency_bins
            mean = mean[electrode_ids] # n_electrodes, 1, n_frequency_bins
            std = std[electrode_ids] # n_electrodes, 1, n_frequency_bins
            # Normalize the data
            electrode_data = (electrode_data - mean) / (std + 1e-5) # n_electrodes, n_timebins, n_frequency_bins    
    else:    
        times = np.arange(electrode_data.shape[1]) / subject.get_sampling_rate()
        frequency_bins = None
    
    if time_start is not None:
        assert time_end is not None, "Time end must be provided if time start is provided"
        times = torch.linspace(time_start, time_end, len(times))

    plot_given_electrode_data(electrode_data, electrode_titles, times, frequencies=frequency_bins, n_columns=n_columns, annotations=annotations, plot_width=plot_width, plot_height=plot_height, save_path=save_path)

def compute_spectrogram_normalization_parameters(subject, trial_id, time_from_seconds=0, time_to_seconds=10*60, laplacian_rereference=True, spectrogram_parameters=None):
    electrode_data = subject.get_all_electrode_data(trial_id, window_from=0, window_to=time_to_seconds * subject.get_sampling_rate())

    if laplacian_rereference:
        from model.preprocessing.laplacian_rereferencing import laplacian_rereference_neural_data
        electrode_data, _, _ = laplacian_rereference_neural_data(electrode_data, subject.get_electrode_labels(), remove_non_laplacian=False)

    from model.preprocessing.spectrogram import SpectrogramPreprocessor
    spectrogram_preprocessor = SpectrogramPreprocessor(spectrogram_parameters=spectrogram_parameters)
    electrode_data, frequency_bins, time_bins = spectrogram_preprocessor({
        'data': electrode_data.unsqueeze(0), 
        'metadata': {
            'sampling_rate': subject.get_sampling_rate()
        }
    }, output_time_frequency_bins=True, z_score=False)

    electrode_data = electrode_data # shape: (1, n_electrodes, n_timebins, n_frequency_bins)
    mean = electrode_data.mean(dim=[0, 2])
    std = electrode_data.std(dim=[0, 2])
    # mean, std shape: (n_electrodes, n_frequency_bins)
    return mean, std

def braintreebank_movie_times_to_neural_index(subject, trial_id, movie_times):
    from evaluation.neuroprobe.config import ROOT_DIR, SAMPLING_RATE

    if isinstance(movie_times, float) or isinstance(movie_times, int):
        movie_times_np = np.array([movie_times])
    elif isinstance(movie_times, list):
        movie_times_np = np.array(movie_times)
    else:
        movie_times_np = movie_times
    assert isinstance(movie_times_np, np.ndarray), "Movie times must be a float, list, or numpy array"

    if type(subject) == int:
        subject_id = subject
    else:
        subject_id = int(subject.subject_id)
    
    # Data frames column IDs
    start_col, end_col = 'start', 'end'
    trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'

    # Path to trigger times csv file
    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{subject_id}_trial{trial_id:03}_timings.csv')

    trigs_df = pd.read_csv(trigger_times_file)
    #display(trigs_df.head())

    last_t = trigs_df[trig_time_col].iloc[-1]
    assert np.all(movie_times_np < last_t), "Movie times must be less than the last trigger time"
    
    # Vectorized nearest trigger finding
    start_indices = np.searchsorted(trigs_df[trig_time_col].values, movie_times_np)
    start_indices = np.maximum(start_indices, 0) # handle the edge case where movie starts right at the word
    
    # Vectorized sample index calculation
    result = np.round(
        trigs_df.loc[start_indices, trig_idx_col].values + 
        (movie_times_np - trigs_df.loc[start_indices, trig_time_col].values) * SAMPLING_RATE
    ).astype(int)

    if isinstance(movie_times, float) or isinstance(movie_times, int):
        return result[0]
    else:
        return result