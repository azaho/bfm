import logging
import time
import psutil
import torch

class ResourceFormatter(logging.Formatter):
    """Formatter displaying time, GPU memory, and RAM usage."""

    def format(self, record: logging.LogRecord) -> str:
        # Time
        current_time = time.strftime("%H:%M:%S")

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_mem = 0.0

        # RAM usage
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024**3

        # Indent support (we store indent on the record if passed)
        indent = getattr(record, "indent", 0)
        indent_str = " " * 4 * indent

        # Core message
        base_msg = super().format(record)
        return f"[{current_time} gpu {gpu_mem:.1f}G ram {ram_usage:.1f}G] {indent_str}{base_msg}"


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with our custom resource formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ResourceFormatter("%(message)s")  # only use msg in body
        handler.setFormatter(formatter)
        logger.addHandler(handler)
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
        return # Kept for backwards compatibility
    logger = get_logger()
    extra = {"indent": indent}
    logger.log(level, message, extra=extra)