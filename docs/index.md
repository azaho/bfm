---
summary: Landing page for the Brain Foundation Models documentation.
authors:
    - Andrii Zahorodnii
date: 2025-08-07
---
# Brain Foundation Models

<p align="center">
  <a href="https://neuroprobe.dev">
    <img src="assets/brain_animation.gif" alt="Neuroprobe Logo" style="height: 10em" />
  </a>
</p>

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-1f425f.svg?color=purple">
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
    </a>
</p>

Welcome to the Brain Foundation Models (BFM) documentation. This site is a work in progress.

## Project layout
```
    docs/
        index.md            # The documentation homepage.
        ...                 # Other markdown pages, images and other files.
    analyses/               # Analysis scripts and notebooks.
    bfm/
        training/
            setups/             # Training setup configurations.
            optimizers/         # Custom optimizers.
        subject/
            datasets/           # Implementation for various datasets
            subjects/           # Implementation for subjects of datasets
        model/
            backbones/          # Model architectures
            encoders/           # Input embedders
            modules/            # Reusable modules
            preprocessing/      # Custom preprocessing functions
        evaluation/
            neuroprobe/         # Neuroprobe benchmark
        core/                   # Core utilities and shared functions
    runs/                  # Output files and logs
    tests/                 # Unit tests and test data
    mkdocs.yml             # Documentation configuration file.
    pyproject.toml         # Python project configuration file.
    quickstart.ipynb       # Quickstart notebook.