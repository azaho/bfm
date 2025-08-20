---
title: Onboarding Task
summary: Task to familiarize new contributors with the Brain Foundation Models project.
authors:
    - Andrii Zahorodnii
date: 2025-08-07
---
# Onboarding Task

To familiarize yourself with the Brain Foundation Models (BFM) project, please try to implement and train a simple linear model. 

This will involve making a copy of the `andrii0.py` file in the `training_setup` directory and modifying that copy. The goal is to get the model to train and evaluate, even if the performance is not optimal. No other files will need to be modified.

**Model Overview:**
The linear model will take the data, average it over all of the electrodes, split into bins of your predetermined size, and then use the linear regression to predict the future bin from the previous bins. The model should be trained with the `L2` loss function.

**Evaluation:**
Feel free to come up with your own scheme of how the model's features (`generate_frozen_features`) will be.
