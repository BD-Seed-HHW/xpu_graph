"""Logger"""
# Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py

import logging
import os
import sys
import threading
from contextlib import contextmanager
from io import StringIO
from typing import Optional


_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


def _get_default_logging_level() -> "logging._Level":
    global _default_log_level

    env_lever_str = os.getenv("SEED_MODELS_LOGGING_LEVEL", None)
    if env_lever_str:
        if env_lever_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[env_lever_str.upper()]
        else:
            raise ValueError(f"Unknown verbosity: {env_lever_str}")

    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "logging.Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    """
    Configures root logger using a stdout stream handler with an explicit format.
    """
    global _default_handler

    with _thread_lock:
        if _default_handler:
            return

        formatter = logging.Formatter(
            fmt="[%(levelname)s][%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    """
    Returns a logger with the specified name. It is not supposed to be accessed by external scripts.
    """
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def set_verbosity_info() -> None:
    """
    Sets the verbosity to the `INFO` level.
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(logging.INFO)


def set_verbosity_warning() -> None:
    """
    Sets the verbosity to the `WARN` level.
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(logging.WARN)


def set_verbosity_debug() -> None:
    """
    Sets the verbosity to the `DEBUG` level.
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(logging.DEBUG)


def set_verbosity_error() -> None:
    """
    Sets the verbosity to the `ERROR` level.
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(logging.ERROR)


@contextmanager
def capture_logger(logger: "logging.Logger") -> "StringIO":
    string_io = StringIO()
    handler = logging.StreamHandler(string_io)
    logger.addHandler(handler)
    yield string_io
    logger.removeHandler(handler)
