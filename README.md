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
    <a href="https://mit-license.org/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://neuroprobe.dev">
        <img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fneuroprobe.dev">
    </a>
</p>

<p align="center"><strong>Brain Foundation Models</strong></p>


## Setup instructions

1. First, create a virtual environment and install the packages:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. If you're not on Openmind, follow the [Neuroprobe repository's](https://github.com/azaho/neuroprobe) instructions for how to download the BrainTreebank dataset and correct the path to the dataset. If you're on Openmind, you are all set!

3. Now you can try pretraining a model! Will require an A100 GPU (see the [openmind.mit.edu](https://openmind.mit.edu) instructions and FAQ for how to request a node with one)
```python
python pretrain.py --training.setup_name andrii0 --cluster.cache_subjects 0 --cluster.eval_at_beginning 0
```