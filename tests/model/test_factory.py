import pytest
import torch
from omegaconf import OmegaConf

from bfm.model.base import BFModule
from bfm.model.factory import build_module
from bfm.model.registry import backbones, encoders


@encoders.register("dummy_enc")
class DummyEncoder(BFModule):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)


@backbones.register("dummy_backbone")
class DummyBackbone(BFModule):
    def __init__(self, dim=10, dummy_list=None):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.dummy_list = dummy_list or []


def make_cfg(**overrides):
    """Returns an OmegaConf DictConfig with attribute access and .get()"""
    base = {
        "encoder": {"registry": "encoders", "name": "dummy_enc", "kwargs": {"dim": 8}},
        "backbone": {
            "registry": "backbones",
            "name": "dummy_backbone",
            "kwargs": {"dim": 8},
        },
    }
    base.update(overrides)
    return OmegaConf.create({"components": base})


def test_build_module_returns_instances():
    cfg = make_cfg()
    enc = build_module("encoder", cfg)
    bb = build_module("backbone", cfg)
    assert isinstance(enc, DummyEncoder)
    assert isinstance(bb, DummyBackbone)
    assert isinstance(enc, BFModule)
    assert isinstance(bb, BFModule)


def test_kwargs_are_passed():
    cfg = make_cfg(
        encoder={"registry": "encoders", "name": "dummy_enc", "kwargs": {"dim": 16}},
        backbone={
            "registry": "backbones",
            "name": "dummy_backbone",
            "kwargs": {
                "dim": 16,
                "dummy_list": [1, 2, 3],  # should be preserved as list
            },
        },
    )
    enc: DummyEncoder = build_module("encoder", cfg)  # type: ignore
    bb: DummyBackbone = build_module("backbone", cfg)  # type: ignore
    assert enc.linear.in_features == 16
    assert bb.linear.out_features == 16
    assert bb.dummy_list == [1, 2, 3]


def test_kwargs_optional():
    # no "kwargs" key → factory should default to {}
    cfg = make_cfg(
        encoder={"registry": "encoders", "name": "dummy_enc"},
        backbone={"registry": "backbones", "name": "dummy_backbone"},
    )
    enc = build_module("encoder", cfg)
    bb = build_module("backbone", cfg)
    assert isinstance(enc, DummyEncoder)
    assert isinstance(bb, DummyBackbone)


def test_missing_component_cfg_raises():
    cfg = make_cfg()
    with pytest.raises(ValueError):
        build_module("missing", cfg)


def test_invalid_registry_name_raises():
    cfg = make_cfg(bad={"registry": "does_not_exist", "name": "x", "kwargs": {}})
    with pytest.raises(ValueError):
        build_module("bad", cfg)


def test_unknown_component_name_raises_keyerror():
    cfg = make_cfg(
        encoder={"registry": "encoders", "name": "does_not_exist", "kwargs": {}}
    )
    with pytest.raises(KeyError):
        build_module("encoder", cfg)


def test_cfg_kwargs_is_not_dict_raises():
    cfg = make_cfg(
        encoder={
            "registry": "encoders",
            "name": "dummy_enc",
            "kwargs": [1, 2, 3],  # should be a dict
        },
        backbone={
            "registry": "backbones",
            "name": "dummy_backbone",
            "kwargs": "not a dict",  # should be a dict
        },
    )

    with pytest.raises(TypeError):
        build_module("encoder", cfg)

    with pytest.raises(ValueError):
        build_module("backbone", cfg)
