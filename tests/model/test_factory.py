import pytest
import torch
from omegaconf import OmegaConf

from bfm.model.base import BFModule
from bfm.model.factory import build_model
from bfm.model.registry import encoders, backbones


@encoders.register("dummy_enc")
class DummyEncoder(BFModule):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)


@backbones.register("dummy_backbone")
class DummyBackbone(BFModule):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)


def make_cfg(**overrides):
    """Returns an OmegaConf DictConfig with attribute access and .get()"""
    base = {
        "encoder":  {"registry": "encoders",  "name": "dummy_enc",      "kwargs": {"dim": 8}},
        "backbone": {"registry": "backbones", "name": "dummy_backbone", "kwargs": {"dim": 8}},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_build_model_adds_submodules():
    cfg = make_cfg()
    model = build_model(module=None, cfg=cfg, components=["encoder", "backbone"])
    assert isinstance(model, BFModule)
    assert isinstance(model.encoder, DummyEncoder)
    assert isinstance(model.backbone, DummyBackbone)


def test_kwargs_are_passed():
    cfg = make_cfg(
        encoder={"registry": "encoders", "name": "dummy_enc", "kwargs": {"dim": 16}},
        backbone={"registry": "backbones", "name": "dummy_backbone", "kwargs": {"dim": 16}},
    )
    model = build_model(None, cfg, ["encoder", "backbone"])
    assert model.encoder.linear.in_features == 16
    assert model.backbone.linear.out_features == 16


def test_kwargs_optional():
    # no "kwargs" key → factory should default to {}
    cfg = make_cfg(
        encoder={"registry": "encoders", "name": "dummy_enc"},
        backbone={"registry": "backbones", "name": "dummy_backbone"},
    )
    model = build_model(None, cfg, ["encoder", "backbone"])
    assert isinstance(model.encoder, DummyEncoder)
    assert isinstance(model.backbone, DummyBackbone)


def test_provided_module_is_used():
    provided = BFModule()
    cfg = make_cfg()
    model = build_model(provided, cfg, ["encoder"])
    assert model is provided
    assert isinstance(model.encoder, DummyEncoder)


def test_missing_component_cfg_raises():
    cfg = make_cfg()
    with pytest.raises(ValueError):
        build_model(None, cfg, ["missing"])


def test_invalid_registry_name_raises():
    cfg = make_cfg(bad={"registry": "does_not_exist", "name": "x", "kwargs": {}})
    with pytest.raises(ValueError):
        build_model(None, cfg, ["bad"])


def test_unknown_component_name_raises_keyerror():
    cfg = make_cfg(encoder={"registry": "encoders", "name": "does_not_exist", "kwargs": {}})
    with pytest.raises(KeyError):
        build_model(None, cfg, ["encoder"])
