---
title: Installation
summary: Instructions for setting up the Brain Foundation Models project codebase.
authors:
    - Andrii Zahorodnii
    - Alexander Brady
date: 2025-08-07
---
# Codebase Installation

If you're not yet invited to join as a collaborator to the ["BFM" GitHub repository](https://github.com/azaho/bfm), please email me at [zaho@mit.edu](mailto:zaho@mit.edu) with your GitHub username. Check your email for an invitation!

## Onboarding

Please see the video at this link for onboarding (requires an MIT zoom log in):
[ONBOARDING VIDEO (20min)](https://mit.zoom.us/rec/share/s2XgwBipwcQDJEmb9OICnecNDenA0EyKidxDg_zP5M9GdvXQxbobaZVtM44AI3fe.4jEyRBNSP2bvQ_cU?startTime=1749952722000)

## Setup Instructions

1. Clone the codebase to Openmind. Create a directory in `/om2/user/<your_username>/<your_name>` and work there.
```sh
git clone https://github.com/azaho/bfm.git
cd bfm
```
2. Create a virtual environment and install the required packages:
```sh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3. If you're not on Openmind, you'll need to download the [BrainTreebank](https://braintreebank.dev/) dataset (~130 GB). If you're on Openmind, you are all set! The dataset is already available in the `/om2/user/<your_username>/braintreebank` directory.
    - Download this script: [braintreebank_download_extract.py](https://github.com/azaho/neuroprobe/blob/main/braintreebank_download_extract.py) from the [Neuroprobe](https://github.com/azaho/neuroprobe) repository.
    - Install necessary packages:
    ```sh
    pip install beautifulsoup4 requests
    ```
    - Run the script to download and extract the dataset:
    ```sh
    python braintreebank_download_extract.py
    ```
    - Update the new path to the dataset in `evaluation/neuroprobe/config.py` and in `subject/braintreebank.py`.

## Weights and Biases

We visualize our training runs using Weights and Biases [(wandb)](https://wandb.ai/site). 

!!! Registration
    Create an account with your educational email address at [wandb.ai](https://wandb.ai/site) . Ensure you indicate that you are a student during registration to get free upgrade. 

Once registered, download the CLI tool:
```sh
pip install wandb 
```

Then, log in to the CLI using your credentials:
```sh
wandb login
```
The token can be found in your [wandb account home](https://wandb.ai/home).

After registration, you can log in to the wandb CLI, pasting your API key when prompted:
```sh
wandb login
``` 

Now, you specify the `wandb_project` in your training script. For example, to set the project name to "bfm":
```sh
python pretrain.py --training.setup_name andrii0 --cluster.wandb_project bfm
```
Now, in the Weights and Biases dashboard, you can see your training runs under the "bfm" project. Projects are similar to folders, and you can call them whatever you like.

## Running a Pretraining Job

Once you have set up the codebase and installed the required packages, you can start pretraining a model. This requires an A100 GPU. If you're on Openmind, you can request a node with an A100 GPU by following the instructions in the [Openmind documentation](https://openmind.mit.edu).
```sh
python pretrain.py --training.setup_name andrii0 --cluster.cache_subjects 0 --cluster.eval_at_beginning 0
```