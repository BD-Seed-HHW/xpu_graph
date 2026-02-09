import functools
import os
from itertools import chain
from typing import Callable

import torch
from torch._dynamo.backends.common import aot_autograd

from .cache import (
    SerializableArtifact,
    SerializableGraphModule,
    XpuGraphCache,
    default_cache,
)
from .config import Target, XpuGraphConfig, get_partition_fn
from .fx_utils import (
    FxStage,
    dispatch_graph,
    fakify_tensors,
    find_hop_nodes,
    find_subclass_inputs,
    freeze_compile,
)
from .passes.pass_manager import PassManager
from .passes.patterns.plugin_pattern import __PLUGIN_PATTERN_GROUP__
from .runtime import XpuGraphRuntimeArtifact
from .utils import GitLikeDiffer, NodesStatistics, local_logger, logger, setup_logger

__all__ = [
    "XpuGraph",
]


class XpuGraph:
    def __init__(
        self,
        config: XpuGraphConfig,
        cache: XpuGraphCache = None,
        **kwargs,
    ):
        config._reset_config_with_env()
        self._config = config
        # Setup logging based on config
        setup_logger(self._config.debug)

        self._cache = cache if cache and config.enable_cache else default_cache() if config.enable_cache else None
        self._set_context()
        # WARNING(liuyuan): _pass_manager MUST be initilized after _set_context because triton kernel depends on environment varaibels that fetched in _set_context.
        self._pass_manager = PassManager(self._config, **kwargs)
        # NOTE(liuyuan): The plugin patterns should be placed before those built-in.
        self._pass_manager.get_pattern_manager().insert_patterns(
            chain.from_iterable(__PLUGIN_PATTERN_GROUP__.get(self._config.target, {}).values())
        )

    def _get_compiler(self, stage: FxStage):
        def _compiler(gm, fake_inputs, stage: FxStage):
            nodes_statistics = NodesStatistics()

            # Create fake inputs for optimization
            fake_mode, fake_inputs = fakify_tensors(fake_inputs)

            with fake_mode:
                if self._config.enable_cache and stage != FxStage.joint:
                    hashkey = self._cache.cache_key(gm, fake_inputs, self._config, stage)
                    cached_compiled = self._cache.load_artifact(hashkey)
                    if cached_compiled is not None:
                        assert isinstance(cached_compiled, SerializableArtifact)
                        return cached_compiled.artifact

                # NOTE(liuyuan): gm could be changed in the compiler, and we should keep the original graph for logging difference.
                original_gm_graph = gm.print_readable(print_output=False, include_stride=True, include_device=True)
                with local_logger("before"):
                    logger.debug(f"before xpu_graph, graph like:\n {original_gm_graph}")
                    logger.info(f"xpu_graph passes start {stage}...")

                nodes_statistics.insert_statistics("before xpu_graph", gm)
                xpu_compiled = self._pass_manager(gm, fake_inputs, stage)
                nodes_statistics.insert_statistics("after xpu_graph", xpu_compiled)

                compiled_gm_graph = xpu_compiled.print_readable(
                    print_output=False, include_stride=True, include_device=True
                )
                with local_logger("after"):
                    logger.info("xpu_graph passes complete")
                    logger.debug(f"after xpu_graph, graph like:\n {compiled_gm_graph}")
                    logger.debug(
                        "Final difference after optimizations by xpu_graph:%s\n",
                        GitLikeDiffer(original_gm_graph, compiled_gm_graph),
                    )

                logger.info(f"node statistic: {str(nodes_statistics)}")

                if (
                    stage in [FxStage.inference, FxStage.forward, FxStage.backward]
                    and self._config.vendor_compiler_config is not None
                ):
                    from .backends import vendor_compiler

                    xpu_compiled = vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        target=self._config.target,
                        is_inference=stage == FxStage.inference,
                        is_backward=stage == FxStage.backward,
                        **self._config.vendor_compiler_config,
                    )
                else:
                    xpu_compiled = SerializableGraphModule(xpu_compiled)

                if self._config.enable_cache and stage != FxStage.joint:
                    if isinstance(xpu_compiled, SerializableArtifact):
                        xpu_compiled = self._cache.save_artifact(hashkey, xpu_compiled)
                    else:
                        logger.warning(f"Artifact of type {type(xpu_compiled)} is not serializable, skip cache.")

                if isinstance(xpu_compiled, SerializableArtifact):
                    # WARNING(liuyuan): MUST get the real artifact itself before return.
                    xpu_compiled = xpu_compiled.artifact

            return xpu_compiled

        def wrapped(gm, sample_inputs):
            with local_logger(stage.name):
                xpu_compiled = _compiler(gm, sample_inputs, stage)
            return xpu_compiled

        return wrapped

    def _legacy_dispatch_and_compile(self, dynamo_gm, example_inputs):
        fallback_dispatch = False
        if self._config.is_training:
            hop_nodes = find_hop_nodes(dynamo_gm)
            if len(hop_nodes) > 0:
                logger.warning(f"Higher order operators detected: {', '.join(str(op) for op in hop_nodes)}.")
                fallback_dispatch = True

        subclass_tensors = find_subclass_inputs(example_inputs)
        if len(subclass_tensors) > 0:
            logger.warning(f"Subclass inputs detected: {', '.join(str(cls) for cls in subclass_tensors)}.")
            fallback_dispatch = True

        if fallback_dispatch:
            logger.debug(
                "before compile: graph like:\n %s",
                dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )
            aot_config = {}
            if self._config.is_training:
                aot_config["fw_compiler"] = self._get_compiler(FxStage.forward)
                aot_config["bw_compiler"] = self._get_compiler(FxStage.backward)
                aot_config["keep_inference_input_mutations"] = True
                if partition_fn := get_partition_fn(self._config.partition_fn):
                    aot_config["partition_fn"] = partition_fn
            else:
                aot_config["fw_compiler"] = self._get_compiler(FxStage.inference)
                aot_config["keep_inference_input_mutations"] = True
            from torch import fx
            def my_partition_fn(
                joint_module: fx.GraphModule,
                *args,
                **kwargs,
                ):
                from xpu_graph.passes.reshard_after_forward import annotate_fsdp_all_gather
                joint_module = annotate_fsdp_all_gather(joint_module, reshard_after_forward=True)
                from torch._functorch.partitioners import default_partition
                return default_partition(joint_module, *args, **kwargs)
            aot_config["partition_fn"] = my_partition_fn
            xpu_gm = aot_autograd(**aot_config)(dynamo_gm, example_inputs)
            fw_metadata = None
        elif self._config.is_training:
            # Since: 1. dynamo has eliminated control-flow for input GraphModule
            #    and 2. aot_autograd traces grad again
            # It's okay use optimized infer-graph for training as well
            logger.debug(
                "before decompose: graph like:\n %s",
                dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )
            logger.info("decompose graph start...")
            dispatched_gm, fake_inputs, fw_metadata = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.pregrad)
            logger.info("decompose graph complete")
            logger.debug(
                "after decompose, graph like:\n %s",
                dispatched_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )

            pregrad_gm = self._get_compiler(FxStage.pregrad)(dispatched_gm, fake_inputs)

            aot_config = {}
            aot_config["fw_compiler"] = self._get_compiler(FxStage.forward)
            aot_config["bw_compiler"] = self._get_compiler(FxStage.backward)
            if partition_fn := get_partition_fn(self._config.partition_fn):
                aot_config["partition_fn"] = partition_fn
            xpu_gm = aot_autograd(**aot_config)(pregrad_gm, fake_inputs)

        else:
            logger.debug(
                "before decompose: graph like:\n %s",
                dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )
            logger.info("decompose graph start...")
            dispatched_gm, fake_inputs, fw_metadata = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.inference)
            logger.info("decompose graph complete")
            logger.debug(
                "after decompose, graph like:\n %s",
                dispatched_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )

            xpu_gm = self._get_compiler(FxStage.inference)(dispatched_gm, fake_inputs)

        return xpu_gm, fw_metadata

    def _dispatch_and_compile(self, dynamo_gm, example_inputs, *args, **kwargs):
        aot_config = {}
        aot_config["keep_inference_input_mutations"] = True
        aot_config["fw_compiler"] = self._get_compiler(FxStage.forward)
        aot_config["bw_compiler"] = self._get_compiler(FxStage.backward)

        def partition_fn(joint_gm, joint_args, *, num_fwd_outputs):
            new_joint_gm = self._get_compiler(FxStage.joint)(joint_gm, joint_args)

            from torch._functorch.partitioners import default_partition

            partition_fn = get_partition_fn(self._config.partition_fn) or default_partition

            return partition_fn(new_joint_gm, joint_args, num_fwd_outputs=num_fwd_outputs)

        aot_config["partition_fn"] = partition_fn
        if self._config.freeze:
            aot_config["inference_compiler"] = functools.partial(
                freeze_compile, dynamo_gm, inner_compiler=self._get_compiler(FxStage.inference)
            )
        else:
            aot_config["inference_compiler"] = self._get_compiler(FxStage.inference)

        compiled = aot_autograd(**aot_config)(dynamo_gm, example_inputs)

        if tracing_context := torch._guards.TracingContext.try_get():
            fw_metadata = tracing_context.fw_metadata
        else:
            fw_metadata = None

        return compiled, fw_metadata

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        logger.info(f"{self._config}")

        if self._config.fallback_legacy_dispatch:
            xpu_gm, fw_metadata = self._legacy_dispatch_and_compile(dynamo_gm, example_inputs)
        else:
            xpu_gm, fw_metadata = self._dispatch_and_compile(dynamo_gm, example_inputs)

        xpu_gm = XpuGraphRuntimeArtifact(xpu_gm)

        if self._config.enable_interceptor is not None:
            from xpu_graph.interceptor import intercept

            logger.info("Wrapping compiled funciton with interceptor")
            xpu_gm = intercept(
                xpu_gm,
                golden_fn=dynamo_gm,
                fw_metadata=fw_metadata,
                is_training=self._config.is_training,
                mark=os.environ.get("RANK", "0"),
                config_str=self._config.enable_interceptor,
            )

        if (guard_filter_fn := kwargs.get("options", {}).get("guard_filter_fn")) is not None and (
            tracing_context := torch._guards.TracingContext.try_get()
        ) is not None:
            orig_guards = tracing_context.guards_context.dynamo_guards
            filter_flags = guard_filter_fn(orig_guards)
            filtered_guards = torch._guards.GuardsSet(
                set(guard for guard, flag in zip(orig_guards, filter_flags) if not flag)
            )
            logger.info(f"Removed guards: {list(filtered_guards)}")
            tracing_context.guards_context.dynamo_guards = orig_guards - filtered_guards

        return xpu_gm

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()

    def _set_context(self):
        self._orig_ctx = {}
        self._orig_ctx["torch._inductor.config.freezing"] = torch._inductor.config.freezing
        if self._config.freeze and self._config.is_training is False:
            # The configuration in this inductor affects the return value of is_parameter_freezing(),
            # thereby influencing the process of generating the fx_graph in dynamo. The current code
            # in the community is not very clean, and it would be more reasonable to place this
            # configuration under dynamo. You can refer to this link for more information.
            # https://github.com/pytorch/pytorch/blob/release/2.5/torch/_dynamo/utils.py#L3061
            torch._inductor.config.freezing = True
        else:
            torch._inductor.config.freezing = False

        if self._config.target != Target.none:
            if torch._dynamo.config.trace_numpy:
                self._orig_ctx["torch._dynamo.config.numpy_default_float"] = torch._dynamo.config.numpy_default_float
                logger.info("xpu_graph set the default traced numpy float dtype to float32")
                torch._dynamo.config.numpy_default_float = "float32"

        if self._cache is not None:
            self._orig_ctx["self._cache.orig_ctx"] = self._cache._set_cache_ctx()

    def _restore_context(self):
        torch._inductor.config.freezing = self._orig_ctx["torch._inductor.config.freezing"]
        if "torch._dynamo.config.numpy_default_float" in self._orig_ctx:
            torch._dynamo.config.numpy_default_float = self._orig_ctx["torch._dynamo.config.numpy_default_float"]
        if self._cache is not None:
            self._cache._restore_cache_ctx(self._orig_ctx["self._cache.orig_ctx"])


class XpuGraphCompiler:
    def __init__(self):
        super().__init__()
        self._config = XpuGraphConfig(is_training=False)
        self._config._reset_config_with_env()
        # for filed in type(self._config).__dataclass_fields__.keys():
        for field in XpuGraphConfig.__dataclass_fields__.keys():
            setter = self._create_setter(field)
            setattr(
                self.__class__,
                field,
                property(self._create_getter(field), setter),
            )
            setattr(self.__class__, "set_%s" % field, self._create_chained_setter(setter))

        self.prior_work()
        self._compiler = None

    def _create_getter(self, field):
        def getter(self):
            return getattr(self._config, field)

        return getter

    def _create_setter(self, field):
        # HACK(liuyuan): typing.Dict[str, typing.Any] is a GenericAlias which is unavailable for isinstance.
        # See https://docs.python.org/3/library/typing.html#typing.Dict and https://docs.python.org/3/library/stdtypes.html#types-genericalias
        expected_type = XpuGraphConfig.__dataclass_fields__[field].type if field != "vendor_compiler_config" else dict

        def setter(self, val):
            if not isinstance(val, expected_type):
                raise TypeError(f"Expected {expected_type}, but got {type(val)}.")
            setattr(self._config, field, val)
            self._compiler = None

        return setter

    def _create_chained_setter(self, setter):
        def chained_setter(self, val):
            setter(self, val)
            return self

        return chained_setter

    def prior_work(self): ...

    def done(self):
        self._compiler = XpuGraph(self._config)

    def compile(self, model: torch.nn.Module, *args, backend=None, **kwargs) -> Callable:
        assert isinstance(
            self._compiler, XpuGraph
        ), "You should call XpuGraphCompiler.done before XpuGraphCompiler.compile"
        return torch.compile(model, backend=self._compiler, *args, **kwargs)
