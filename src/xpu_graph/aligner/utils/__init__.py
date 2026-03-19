import importlib
import logging
logger = logging.getLogger("xpu_graph")

__all__ = []

if importlib.util.find_spec("mariana") is not None:
    from .mariana_utils import XpuGraphAlignerMarianaCallback
    __all__.append("XpuGraphAlignerMarianaCallback")
    logger.info("[ALIGNER] XpuGraphAlignerMarianaCallback is available.")
