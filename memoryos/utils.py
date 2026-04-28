"""
Logging utilities for MemoryOS.
"""

import logging
from typing import Union


def get_logger(name: str, level: Union[str, int] = "INFO") -> logging.Logger:
    """
    Return a consistently formatted logger for the given name.

    Loggers are cached by Python's logging module, so calling this multiple
    times with the same ``name`` is safe and cheap.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level as a string (``"DEBUG"``, ``"INFO"``, etc.) or
            an ``int`` from the :py:mod:`logging` constants.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  [MemoryOS] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    return logger
