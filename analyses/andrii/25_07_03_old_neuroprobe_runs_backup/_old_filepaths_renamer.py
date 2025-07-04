import os

for split_type in ['SS_SM', 'SS_DM', 'DS_DM']:
    models = [
        {
            'name': 'Linear',
            'color_palette': 'viridis',
            'eval_results_path': f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_remove_line_noise/'
        },
        {
            'name': 'Linear (STFT)',
            'color_palette': 'viridis', 
            'eval_results_path': f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_realimag/'
        },
        {
            'name': 'Linear (spectrogram)',
            'color_palette': 'viridis',
            'eval_results_path': f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_abs/'
        },
        {
            'name': 'Transformer (raw voltage)',
            'color_palette': 'rainbow',
            'eval_results_path': f'/om2/user/zaho/bfm/runs/analyses/andrii/25_07_03_old_neuroprobe/eval_results_lite_{split_type}/transformer_voltage/'
        },
        {
            'name': 'CNN (raw voltage)',
            'color_palette': 'rainbow',
            'eval_results_path': f'/om2/user/zaho/bfm/runs/analyses/andrii/25_07_03_old_neuroprobe/eval_results_lite_{split_type}/cnn_voltage/'
        },
        {
            'name': 'BrainBERT',
            'color_palette': 'plasma',
            'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_lite_{split_type}/brainbert_frozen_mean_granularity_{-1}/'
        },
        # {
        #     'name': 'PopT (frozen)',
        #     'color_palette': 'magma',
        #     'eval_results_path': f'/om2/user/zaho/btbench/eval_results_popt/population_frozen_{split_type}_results.csv'
        # },
        {
            'name': 'PopT',
            'color_palette': 'magma',
            'eval_results_path': f'/om2/user/zaho/btbench/eval_results_popt/popt_{split_type}_results.csv'
        }
    ]
    for model in models:
        target_base = '/om2/user/zaho/bfm/runs/analyses/andrii/25_07_03_old_neuroprobe/'
        os.makedirs(target_base, exist_ok=True)
        
        source_path = model['eval_results_path']
        
        if os.path.isfile(source_path):
            # If source is a file, copy the file
            target_path = os.path.join(target_base, os.path.basename(source_path))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            os.system(f'cp {source_path} {target_path}')
            print(f'cp {source_path} {target_path}')
        else:
            # If source is a directory, copy the directory and contents
            # For directories, we want to copy the contents to the target_base directly
            # Remove trailing slash and get the basename
            source_dir_name = os.path.basename(source_path.rstrip('/'))
            target_path = target_base#sos.path.join(target_base, source_dir_name)
            os.system(f'cp -r {source_path} {target_path}')
            print(f'cp -r {source_path} {target_path}')