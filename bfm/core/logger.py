import logging
import time
from typing import Optional

import psutil
import torch


class ResourceFormatter(logging.Formatter):
    """
    Formatter showing timestamp, level, origin, GPU/RAM, indent, and message.

    Custom record extras:
      - indent: int  (levels of 4 spaces)
      - local:  bool (if True, suppress [name:lineno])
    """

    def format(self, record: logging.LogRecord) -> str:
        t = time.strftime("%H:%M:%S")
        lvl = record.levelname  # DEBUG, INFO, WARNING, ERROR, CRITICAL

        local = getattr(record, "local", False)
        where = f"{record.name}:{record.lineno} - " if not local else ""

        if torch.cuda.is_available():
            d = torch.cuda.current_device()
            a = torch.cuda.memory_allocated(d) / 2**30
            rsv = torch.cuda.memory_reserved(d) / 2**30
            gpu = f"cuda:{d} {a:.1f}/{rsv:.1f}G"
        else:
            gpu = "cpu"

        ram = psutil.Process().memory_info().rss / 2**30
        indent = " " * (4 * int(getattr(record, "indent", 0)))
        msg = super().format(record)  # "%(message)s"
        return f"[{t} {lvl}][{gpu}][RAM {ram:.1f}G] {indent}{where} {msg}"


def get_logger(name: str = __name__, level: Optional[int] = None) -> logging.Logger:
    """Create a logger with our custom resource formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ResourceFormatter("%(message)s")  # only use msg in body
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    if level is not None:
        logger.setLevel(level)
        
    return logger


def log(message: str, level: int = logging.INFO, indent: int = 0, priority: int = 0):
    """
    Convenience wrapper for quick logging.

    Args:
        message (str): Log message.
        level (int): Logging level (default INFO).
        indent (int): Indentation levels (4 spaces each).
        priority (int): Logging priority (default 0). If greater than 1, don't log.
    """
    if priority > 1:
        return  # Kept for backwards compatibility
    logger = get_logger()
    extra = {"indent": indent, "local": True}
    logger.log(level, message, extra=extra)
