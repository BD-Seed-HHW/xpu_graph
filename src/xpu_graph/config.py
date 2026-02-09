import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Any, Callable, Dict, List, Optional, Union

from .utils import __XPU_GRAPH_ENVS__, get_bool_env_var, logger


class Target(Enum):
    mlu = "mlu"
    none = "none"
    npu = "npu"


@total_ordering
class OptLevel(Enum):
    level0 = 0  # Close all optimizer
    level1 = 1  # Reture results with bitwise alignment
    level2 = 2  # No guarantee fot bitwise alignment
    level3 = 3  # Placeholder, same with level2 now

    def __lt__(self, other):
        if isinstance(other, OptLevel):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, OptLevel):
            return self.value == other.value
        return NotImplemented


@dataclass
class XpuGraphConfig:
    """Configuration for XPU graph execution."""

    is_training: bool  # Must fill, if is_training is True, XpuGraph will work as a training compiler, otherwise a inference compiler
    debug: bool = False
    target: Target = field(default_factory=lambda: Target.none)  # Target hardware backend
    opt_level: OptLevel = field(default_factory=lambda: OptLevel.level1)
    dump_graph: bool = False
    enable_cache: bool = True
    freeze: bool = (
        # Only take effects when "is_training" is False.
        # Freezing parameter will change model's parameter from inputs into attributes.
        # This may help XpuGraph do better constant folding.
        # WARNING: DO NOT freeze if you need to UPDATE your parameters
        False
    )
    constant_folding: bool = True
    folding_freezed_params: bool = (
        # (Experimental) Only take effects whe freeze is True and constant_folding is True
        # When freeze is True, params exists as attributes in GraphModule.
        # If folding_freezed_params is True, XpuGraph will treat freezed parameters as constants and fold them
        # If folding_freezed_params is False, XpuGraph will not fold freezed parameters and allow parameter hot-swapping
        True
    )
    bucketing_and_reordering: bool = False

    # So far we only support configure "mode", because we mainly use "Inductor" as a vendor's compiler.
    # mode must be one of {"cudagraphs", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"},
    # we add a "cudagraphs" option. At this mode, XpuGraph will only enable torch.compile in-tree backend "cudugraphs".
    # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
    vendor_compiler_config: Optional[Dict[str, Any]] = None

    # Users can enable interceptor to monitor the results of compiled graph
    enable_interceptor: Optional[str] = None

    # Users can specify which patterns can be included
    include_patterns: List[str] = field(default_factory=list)

    # Users can specify which patterns can be skipped
    skip_patterns: List[str] = field(default_factory=list)

    # Whether to use legacy dispatchers in case of higher-order operators or subclass-tensors
    fallback_legacy_dispatch: bool = True

    # Users can specify which partition function to use for training
    partition_fn: Optional[Union[Callable, str]] = None

    def _reset_config_with_env(self):
        import os

        if os.getenv(__XPU_GRAPH_ENVS__.debug) is not None:
            self.debug = get_bool_env_var(__XPU_GRAPH_ENVS__.debug, False)

        opt_level_env = os.getenv(__XPU_GRAPH_ENVS__.opt_level, str(self.opt_level.value))
        if opt_level_env == "0":
            self.opt_level = OptLevel.level0
        elif opt_level_env == "1":
            self.opt_level = OptLevel.level1
        elif opt_level_env == "2":
            self.opt_level = OptLevel.level2
        elif opt_level_env == "3":
            self.opt_level = OptLevel.level3
        else:
            warnings.warn("Illegal XPUGRAPH_OPT_LEVEL value, XPUGRAPH_OPT_LEVEL will not take effect.")

        vendor_compiler_mode = os.getenv(__XPU_GRAPH_ENVS__.vendor_compiler_mode, "Null")
        if vendor_compiler_mode != "Null":
            if vendor_compiler_mode == "none":
                self.vendor_compiler_config = None
            else:
                if vendor_compiler_mode not in (
                    "default",
                    "cudagraphs",
                    "reduce-overhead",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                ):
                    warnings.warn("Illegal VENDOR_COMPILER_MODE value, VENDOR_COMPILER_MODE will not take effect.")
                else:
                    self.vendor_compiler_config = {"mode": vendor_compiler_mode}

        if os.getenv(__XPU_GRAPH_ENVS__.enable_interceptor) is not None:
            self.enable_interceptor = os.getenv(__XPU_GRAPH_ENVS__.enable_interceptor)

        if os.getenv(__XPU_GRAPH_ENVS__.include_patterns) is not None:
            self.include_patterns = os.getenv(__XPU_GRAPH_ENVS__.include_patterns).split(",")

        if os.getenv(__XPU_GRAPH_ENVS__.skip_patterns) is not None:
            self.skip_patterns = os.getenv(__XPU_GRAPH_ENVS__.skip_patterns).split(",")

        if os.getenv(__XPU_GRAPH_ENVS__.fallback_legacy_dispatch) is not None:
            self.fallback_legacy_dispatch = get_bool_env_var(
                __XPU_GRAPH_ENVS__.fallback_legacy_dispatch, self.fallback_legacy_dispatch
            )

        if os.getenv(__XPU_GRAPH_ENVS__.partition_fn) is not None:
            self.partition_fn = get_partition_fn(os.getenv(__XPU_GRAPH_ENVS__.partition_fn))


cache_path = os.getenv(__XPU_GRAPH_ENVS__.cache_dir)


def get_cache_dir():
    global cache_path
    if cache_path is None:
        import tempfile

        cache_path = tempfile.mkdtemp(prefix="xpugraph_")
        os.environ[__XPU_GRAPH_ENVS__.cache_dir] = cache_path
        logger.debug(f"Use {cache_path} as default local cache")
    return cache_path


dump_path = os.getenv(__XPU_GRAPH_ENVS__.dump_dir)


def get_dump_dir():
    global dump_path
    if dump_path is None:
        import tempfile

        dump_path = tempfile.mkdtemp(prefix="xpugraph_")
        os.environ[__XPU_GRAPH_ENVS__.dump_dir] = dump_path
        logger.debug(f"Use {dump_path} as default dump path")
    return dump_path


def get_partition_fn(fqn):
    if not isinstance(fqn, str):
        return fqn
    from torch._functorch.partitioners import (
        default_partition,
        min_cut_rematerialization_partition,
    )

    if fqn.lower() == "default":
        return default_partition
    elif fqn.lower() == "mincut":
        return min_cut_rematerialization_partition
    else:
        try:
            import importlib

            mod_name, func_name = fqn.rsplit(".", 1)
            module = importlib.import_module(mod_name)
            func = getattr(module, func_name)
            assert isinstance(func, Callable), f"Partition function {fqn} is not a callable object"
            return func
        except Exception:
            warnings.warn(f"Failed to import partition function {fqn}, use default partition function")
            return None
