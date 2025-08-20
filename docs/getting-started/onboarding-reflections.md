---
title: Onboarding
summary: Reflections on the onboarding process for new contributors to the Brain Foundation Models project. 
authors:
    - Alexander Brady
date: 2025-08-07
---
# My Onboarding Reflections

This page covers my experience with the onboarding task for the Brain Foundation Models (BFM) project. It can also serves as a guide for future contributors to understand the process and expectations.

## Initial Impressions

After exploring the codebase, I found it well-structured and easy to navigate. As I'm using ETH Zurich's Euler cluster, I had to adapt the setup instructions slightly to fit my environment. The provided instructions were clear, but I had to ensure that the paths and configurations matched my system. I also defined my own `.run.sh` script to handle the SLURM job submission.

## Implementation Steps

I began by copying the `andrii0.py` file to `bradya0.py` and modifying it to suit my needs. I first removed the model components such as the `OriginalModel` and `TimeTransformer`, as I was going to build a much simpler linear model. For this, I defined a new `LinearModel` class that inherits from `BFMModule`. 

**Linear Model Implementation:**

The `__init__` method initializes the model with a linear layer, and the `forward` method processes the input data. Following the instructions, the model's forward method performs the following steps:

- The model averages the electrode data across the first (`n_electrodes`) dimension.
- These averaged features are then binned into a specified number of bins (I used `n_bins = 10`).
- Each bin is used as input to the linear layer, which outputs a prediction for the next bin.
- Finally, the model returns two tensors, where at point `i`, the first tensor contains the prediction for bin `i` and the second tensor contains the data for bin `i+1`.

**Training Setup:**

I kept my training setup quite simple. The `model_components['model']` was set to an instance of `LinearModel`, and I kept the `model_components['electrode_embeddings']` to `None`, as I didn't use any embeddings in my linear model. 

I also removed the additional `preprocess` steps that were not necessary for this model.

**Pretraining Loss**

As per the instructions, I utilized the `L2Loss` for the pretraining task. This loss function was implemented with the `torch.nn.MSELoss` class, which computes the mean squared error between the predicted and actual values for the next bin.

**Evaluation**

I used a very simple evaluation setup for the `generate_frozen_features` method. I simply took the model's predictions and averaged them across the first dimension (the bins).

**Results**

The model began with an initial L2 loss of around 1200 to a final batch loss of 6.7 after 100 epochs. By 20 epochs, the loss had already dropped to around 20, and it took 70 epochs to reach a loss of under 7. 