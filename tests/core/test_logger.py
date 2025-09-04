import io
import logging
import types

import torch

from bfm.core import logger as logger_mod


def make_captured_logger(name="test.logger", level=logging.INFO):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logger_mod.ResourceFormatter("%(message)s"))
    lg = logging.getLogger(name)
    # Clean slate for this named logger
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(level)
    lg.addHandler(handler)
    return lg, stream


def test_get_logger_adds_one_handler():
    lg1 = logger_mod.get_logger(name="test.singleton", level=logging.INFO)
    lg2 = logger_mod.get_logger(name="test.singleton", level=logging.INFO)
    assert lg1 is lg2
    # Should not accumulate handlers on repeated calls
    assert len(lg1.handlers) == 1


def test_log_writes_message(monkeypatch):
    # Route logger_mod.log() to a controlled logger with a StringIO
    lg, stream = make_captured_logger(name="test.capture", level=logging.INFO)
    monkeypatch.setattr(logger_mod, "get_logger", lambda: lg)

    logger_mod.log("hello world", indent=2)
    out = stream.getvalue()
    assert "hello world" in out  # don't assert exact formatting


def test_log_priority_suppresses_output(monkeypatch):
    lg, stream = make_captured_logger(name="test.priority", level=logging.INFO)
    monkeypatch.setattr(logger_mod, "get_logger", lambda: lg)

    logger_mod.log("should not appear", priority=2)
    assert stream.getvalue() == ""


def test_logger_level_respected(monkeypatch):
    lg, stream = make_captured_logger(name="test.level", level=logging.INFO)
    monkeypatch.setattr(logger_mod, "get_logger", lambda: lg)

    logger_mod.log("debug msg", level=logging.DEBUG)
    assert stream.getvalue() == ""  # DEBUG should be filtered at INFO level


def test_resourceformatter_cpu_path(monkeypatch):
    # Ensure the CPU branch runs even if CUDA is present in the environment
    # We only assert message presence, not the format details.
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(logger_mod.torch, "cuda", fake_cuda, raising=True)

    lg, stream = make_captured_logger(name="test.cpu", level=logging.INFO)
    lg.info("msg via cpu path", extra={"local": True, "indent": 1})

    out = stream.getvalue()
    assert "msg via cpu path" in out


def test_resourceformatter_gpu_simulated(monkeypatch):
    # Fake cuda API
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 1024**3)  # 1 GB
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 2 * 1024**3)  # 2 GB

    lg, stream = make_captured_logger("test.gpu.sim", level=logging.INFO)
    lg.info("msg simulated gpu", extra={"local": True})
    out = stream.getvalue()

    assert "msg simulated gpu" in out
    assert "cuda:0" in out
    assert "1.0" in out  # Memory allocated
    assert "2.0" in out  # Memory reserved
