import dataclasses
import warnings
from os import PathLike
from typing import Any, Dict


def _check_torch():
    REQUIRED_MIN = "2.7.0a0"
    REQUIRED_MAX = "2.8.0"

    try:
        import torch
    except ImportError:
        raise ImportError(
            f"xpu_graph requires PyTorch (>={REQUIRED_MIN},<{REQUIRED_MAX}) "
            "but torch is not installed in the current environment. "
            "Please install a compatible version manually: "
            "https://pytorch.org/get-started/locally/"
        )

    from packaging.version import Version

    ver = Version(torch.__version__.split("+")[0])
    if not (Version(REQUIRED_MIN) <= ver < Version(REQUIRED_MAX)):
        warnings.warn(
            f"xpu_graph is designed for torch>={REQUIRED_MIN},<{REQUIRED_MAX}, "
            f"but the current environment has torch=={torch.__version__}. "
            "Some features may not work correctly.",
            stacklevel=2,
        )


_check_torch()

from .cache import XpuGraphCache, XpuGraphLocalCache, default_cache, no_cache
from .compiler import XpuGraph
from .config import OptLevel, Target, XpuGraphConfig
from .device_graph_runner import GraphRunner
from .guard_filters import *
from .passes.patterns.plugin_pattern import *
from .version import __version__

__all__ = [
    "XpuGraph",
    "XpuGraphConfig",
    "Target",
    "OptLevel",
    "XpuGraphCache",
    "default_cache",
    "mlu_compiler",
    "npu_compiler",
    "register_plugin_pattern",
    "register_this_as_plugin_pattern",
    "register_this_as_pattern_constraint",
    "deregister_plugin_patterns",
    "enable_plugin_patterns",
    "skip_all_guards_unsafe",
    "GraphRunner",
]


def npu_compiler(
    freeze: bool = False,
    opt_level: OptLevel = OptLevel.level1,
    constant_folding: bool = True,
    debug: bool = False,
    **kwargs,
):
    # Note:
    #   By default, npu_compiler uses empty config to invoke vendor_compiler,
    #   It is equivalent to vendor_compiler_config={"compiler": "ge", "mode": "reduce-overhead"} , using aclgraph as the backend
    #   If you want to use inductor, you need to set vendor_compiler_config={"compiler": "inductor"} and the default mode is "default"
    vendor_compiler_config: Dict[str, Any] = kwargs.get("vendor_compiler_config", {})

    if "cache" in kwargs:
        cache: XpuGraphCache = kwargs["cache"]
    else:
        cache: XpuGraphCache = default_cache()

    config = XpuGraphConfig(
        is_training=False,
        target=Target.npu,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
        debug=debug,
        vendor_compiler_config=vendor_compiler_config,
    )
    return XpuGraph(config, cache)


def mlu_compiler(
    is_training: bool,
    **patch_configs,
) -> XpuGraph:
    """
    Create an MLU compiler configuration and return an XpuGraph instance.

    Possible Patch Args:
        freeze: Whether to freeze the graph.
        opt_level: Optimization level.
        constant_folding: Whether to enable constant folding.
        cache: Cache for compiled graphs. Uses default cache if None.
        debug: Whether to enable debug mode.
        vendor_compiler_config: Additional vendor-specific compiler configuration.

    Returns:
        An XpuGraph instance configured for MLU.
    """
    assert "target" not in patch_configs, "target is not allowed to be set here"

    if "cache" not in patch_configs:
        cache = no_cache() if is_training else default_cache()
    elif isinstance(patch_configs["cache"], (str, PathLike)):
        cache = XpuGraphLocalCache(patch_configs["cache"])
    else:
        assert isinstance(patch_configs["cache"], XpuGraphCache), "cache must be a XpuGraphCache instance"
        cache = patch_configs["cache"]

    default_config = _MLU_TRAIN_CONFIG if is_training else _MLU_INFER_CONFIG

    if "opt_level" in patch_configs and not isinstance(patch_configs["opt_level"], OptLevel):
        patch_configs["opt_level"] = OptLevel(int(patch_configs["opt_level"]))
    patch_configs = {
        field.name: patch_configs[field.name]
        for field in dataclasses.fields(XpuGraphConfig)
        if field.name in patch_configs
    }
    config = dataclasses.replace(default_config, **patch_configs)

    if not is_training:
        import torch_mlu_ops

    if is_training:
        return XpuGraph(config, cache)
    else:
        # Note: this is a legacy patch, as some outer framework requires the xpugraph-returned compiled artifact to be pickable
        class PatchedPickableXpuGraphBackend(XpuGraph):
            def __call__(self, *args, **kwargs):
                compiled = super().__call__(*args, **kwargs)
                from torch._inductor.compile_fx import CompiledFxGraph
                from torch.fx import GraphModule

                from .cache import SerializableCompiledFxGraph, SerializableGraphModule
                from .runtime import XpuGraphRuntimeArtifact

                if isinstance(compiled, XpuGraphRuntimeArtifact):
                    if isinstance(compiled._compiled_func, GraphModule):
                        compiled._compiled_func = SerializableGraphModule(compiled._compiled_func)
                    elif isinstance(compiled._compiled_func, CompiledFxGraph):
                        compiled._compiled_func = SerializableCompiledFxGraph(compiled._compiled_func)
                    else:
                        return compiled._compiled_func

                    return compiled.__call__

                return compiled

        return PatchedPickableXpuGraphBackend(config, cache)


_MLU_TRAIN_CONFIG = XpuGraphConfig(
    is_training=True,
    debug=False,
    target=Target.mlu,
    enable_cache=True,
    freeze=False,
    opt_level=OptLevel.level2,
    constant_folding=False,
    vendor_compiler_config={"mode": "default"},
    fallback_legacy_dispatch=True,
)

_MLU_INFER_CONFIG = XpuGraphConfig(
    is_training=False,
    debug=False,
    target=Target.mlu,
    enable_cache=True,
    freeze=False,
    opt_level=OptLevel.level2,
    constant_folding=False,
    vendor_compiler_config={"mode": "reduce-overhead"},
    fallback_legacy_dispatch=True,
)
