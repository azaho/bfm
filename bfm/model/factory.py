from typing import Tuple
from .base import BFModule
from .registry import encoders, backbones


def build_model(cfg) -> Tuple[BFModule, BFModule]:
    """
    Build the model components (encoder and backbone) based on the configuration.
    
    Args:
        cfg: The configuration object containing model parameters.
            Should include:
                - device (str), model.dtype (torch.dtype)
                - encoder.name (str), encoder.kwargs (dict)
                - backbone.name (str), backbone.kwargs (dict)

    Returns:
        Tuple(BFModule, BFModule): The encoder and backbone modules.
    """
    encoder = encoders.resolve(cfg.encoder.name, **cfg.encoder.kwargs)
    backbone = backbones.resolve(cfg.backbone.name, **cfg.backbone.kwargs)

    encoder.to(cfg.device, cfg.model.dtype)
    backbone.to(cfg.device, cfg.model.dtype)
    
    return encoder, backbone