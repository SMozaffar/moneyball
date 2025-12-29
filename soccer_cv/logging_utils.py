from __future__ import annotations

from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()

def get_logger(name: str = "soccer_cv", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = RichHandler(rich_tracebacks=True, console=console, show_time=True, show_level=True, show_path=False)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
