# bfm_ic

1. First, create a virtual environment and install the packages:
```
python -m venv .venv
source .venv/bin/activate
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn psutil ml_dtypes plotly nbformat mne pyedflib moabb pynwb dandi
```

2. Clone btbench from https://github.com/azaho/btbench
```
git clone https://github.com/azaho/btbench.git
```

3. Follow the instructions in the btbench README to download the BrainTreebank data and put it in the correct directory, updating the `btbench_config.py` file with the correct path.

4. Run the script `btbench_process_subject_trial_df.py` in btbench to process the subject trial data for benchmarking and save it to `btbench_subject_metadata` directory.

5. Now, you can run the script `train_model.py` to train the model. (use `run_train_model.sh` to run it on the cluster.)