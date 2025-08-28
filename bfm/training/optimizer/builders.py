from typing import Mapping, List

import torch
from torch import optim
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, SequentialLR, _LRScheduler
)

from .muon import Muon


def build_optimizers(
    model_params: List[torch.nn.Parameter], config: Mapping
) -> List[optim.Optimizer]:
    """
    Build a list of optimizers based on the model parameters and configuration.

    Args:
        model_params (List[torch.nn.Parameter]): List of model parameters to optimize.
        config (Mapping): Configuration dictionary containing optimizer settings.

    Returns:
        List[optim.Optimizer]: List of optimizers.
    """
    params = [p for p in model_params if p.requires_grad and p.is_floating_point()]
    optimizers: List[optim.Optimizer] = []

    use_muon = config["training"]["optimizer"] == "Muon"
    matrix_params = [p for p in params if p.ndim == 2] if use_muon else []
    other_params = [p for p in params if p.ndim != 2] if use_muon else params

    if use_muon and matrix_params:  # Muon only supports matrix parameters
        optimizers.append(
            Muon(
                matrix_params,
                lr=config["training"]["learning_rate"],
                momentum=0.95,
                nesterov=True,
                backend="newtonschulz5",
                backend_steps=5,
                weight_decay=config["training"]["weight_decay"],
            )
        )

    if other_params:  # covers: non-Muon path, or Muon’s leftovers, or Muon→fallback
        optimizers.append(
            optim.AdamW(
                other_params,
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
                betas=(0.9, 0.95),
            )
        )

    return optimizers


def build_schedulers(
    optimizers: List[optim.Optimizer], config: Mapping, training_setup
) -> List[_LRScheduler]:
    """
    Build learning rate schedulers for the given optimizers.
    Both warmup and falloff schedules are supported (both optional).

    Args:
        optimizers (optim.Optimizer): List of optimizers to build schedulers for.
        config (Mapping): Configuration dictionary containing scheduler settings.
        training_setup: Training setup object containing dataloaders.

    Returns:
        List[_LRScheduler]: List of learning rate schedulers.
    """
    schedulers: List[_LRScheduler] = []
    total_steps = config["training"]["n_epochs"] * len(
        training_setup.train_dataloader
    )
    
    for optimizer in optimizers:
        # Warmup schedule
        if config["training"]["warmup_steps"] > 0:
            warmup = LinearLR(
                        optimizer,
                        start_factor=1e-5,
                        end_factor=1.0,
                        total_iters=config["training"]["warmup_steps"],
                )
        else:
            warmup = None

        # Main schedule
        if config["training"]["lr_schedule"] == "linear":
            main = LinearLR(
                optimizer, start_factor=1.0, end_factor=1e-5, total_iters=total_steps
            )
        elif config["training"]["lr_schedule"] == "cosine":
            main = CosineAnnealingLR(optimizer, T_max=total_steps)
        else:
            main = None

        if warmup and main:
            schedulers.append(
                SequentialLR(
                    optimizer,
                    [warmup, main], 
                    milestones=[config["training"]["warmup_steps"]]
                )
            )
        elif warmup:
            schedulers.append(warmup)
        elif main:
            schedulers.append(main)
            
    return schedulers