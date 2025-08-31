import torch
import pytest

from bfm.model.factory import build_model
from bfm.model.base import BFModule
from bfm.model.registry import encoders, backbones


@encoders.register("dummy_enc")
class DummyEncoder(BFModule):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


@backbones.register("dummy_backbone")
class DummyBackbone(BFModule):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class DummyCfg:
    """Mimics Hydra/OmegaConf config tree with attributes."""

    class encoder:
        name = "dummy_enc"
        kwargs = {"dim": 8}

    class backbone:
        name = "dummy_backbone"
        kwargs = {"dim": 8}

    device = "cpu"

    class model:
        dtype = torch.float32


def test_build_model_returns_modules():
    cfg = DummyCfg()
    enc, bbk = build_model(cfg)
    assert isinstance(enc, BFModule)
    assert isinstance(bbk, BFModule)


def test_modules_on_correct_device_and_dtype():
    cfg = DummyCfg()
    cfg.device = "cpu"
    cfg.model.dtype = torch.float32
    enc, bbk = build_model(cfg)

    # check parameter device and dtype
    p = next(enc.parameters())
    assert str(p.device) == cfg.device
    assert p.dtype == cfg.model.dtype

    p = next(bbk.parameters())
    assert str(p.device) == cfg.device
    assert p.dtype == cfg.model.dtype


def test_kwargs_are_passed():
    cfg = DummyCfg()
    cfg.encoder.kwargs = {"dim": 16}
    cfg.backbone.kwargs = {"dim": 16}
    enc, bbk = build_model(cfg)

    # both should have Linear layers with in/out = 16
    assert enc.linear.in_features == 16
    assert bbk.linear.out_features == 16


def test_invalid_name_raises():
    cfg = DummyCfg()
    cfg.encoder.name = "does_not_exist"
    with pytest.raises(KeyError):
        build_model(cfg)

    cfg = DummyCfg()
    cfg.backbone.name = "does_not_exist"
    with pytest.raises(KeyError):
        build_model(cfg)
