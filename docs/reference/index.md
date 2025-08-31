---
title: Reference
summary: A brief overview of the project's codebase and its components.
authors:
    - Alexander Brady
date: 2025-07-15
---
# Brain Foundation Models Code Reference

This section provides detailed documentation of the codebase. Use the navigation on the left to explore different components of the project.


**Source Code Layout**
```
    bfm/
        training/
            setups/             # Training setup configurations.
            optimizers/         # Custom optimizers.
        subject/
            datasets/           # Implementation for various datasets
            subjects/           # Implementation for subjects of datasets
        model/
            encoders/           # Input embedders
            backbones/          # Model architectures
            modules/            # Reusable modules
            preprocessing/      # Custom preprocessing functions
        evaluation/
            neuroprobe/         # Neuroprobe benchmark
        core/                   # Core utilities and shared functions
```