import pickle
from typing import Dict

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import GraphModule

from xpu_graph.backends import device_graph
from xpu_graph.cache import SerializableArtifact, temp_disable_tracing_envs
from xpu_graph.utils import logger, recursive_set_obj


class NpuSerializableArtifact(SerializableArtifact):
    def __init__(self, artifact):
        assert hasattr(artifact, "dump_artifacts")
        super().__init__(artifact)

    def convert_to_bytes(self):
        # NOTE(liuyuan): Since tng_backend does not save any tenosr, would it be necessary?
        with temp_disable_tracing_envs():
            return pickle.dumps(self._artifact.dump_artifacts())

    @staticmethod
    def rebuild_from_bytes(bytes):
        from torchair.npu_fx_compiler import _CompiledFxGraph

        # NOTE(liuyuan): Since tng_backend does not save any tenosr, would it be necessary?
        with temp_disable_tracing_envs():
            return __class__(_CompiledFxGraph.load_artifacts(pickle.loads(bytes)))


def has_triton_kernel(gm: GraphModule):
    for node in gm.graph.nodes:
        if node.op == "call_function" and getattr(node.target, "namespace", "dummy") == "torch_npu_triton":
            return True
    return False


def ge_compiler(module: torch.nn.Module, example_inputs, **config_dict: Dict) -> torch.nn.Module:

    torch.npu.set_compile_mode(jit_compile=False)

    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    recursive_set_obj(config_dict, config)
    if (
        mode := config_dict.get(
            "mode",
            (
                "max-autotune" if "compiler" in config_dict else "reduce-overhead"
            ),  # NOTE(liuyuan): If user specify the compiler, then we should consider it as GE instead of AclGraph.
        )
    ) == "reduce-overhead":
        config.mode = mode
        from torch import SymInt

        for ele in example_inputs:
            if isinstance(ele, SymInt):
                raise TypeError("ACL Graph does not support dynamic shape!!")

        if mempool := config_dict.get("use_custom_pool", None):
            config.aclgraph_config.use_custom_pool = mempool

    npu_backend = tng.get_compiler(compiler_config=config)

    from torchair._utils import get_npu_default_decompositions

    module = make_fx(
        module,
        decomposition_table=get_npu_default_decompositions(),
        tracing_mode="fake",
        record_module_stack=True,
    )(*example_inputs)

    compiled_module = npu_backend(module, example_inputs)

    if not has_triton_kernel(module):
        compiled_module = NpuSerializableArtifact(compiled_module)
    else:
        logger.info("Triton kernel found in gm, skip serializing npu artifact.")

    return compiled_module


def inductor_compiler(
    module: torch.nn.Module,
    inputs,
    *,
    is_inference: bool = False,
    is_backward: bool = False,
    **config_dict: Dict,
) -> torch.nn.Module:
    logger.info("Decompose gm for npu_inductor")
    from xpu_graph.fx_utils import decompose_for_inductor

    module = decompose_for_inductor(module, inputs)
    logger.debug(
        "After decompose_for_inductor, graph like:\n %s",
        module.print_readable(print_output=False, include_stride=True, include_device=True),
    )

    from torch import _TorchCompileInductorWrapper
    from torch._inductor.compile_fx import compile_fx_inner

    # default means None. In torch, _TorchCompileInductorWrapper's apply_mode just passes.
    mode = config_dict.get("mode", "default")
    options = {}
    dynamic = config_dict.get("dynamic", False)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)

    with torch._inductor.config.patch(inductor_backend.config):
        compiled_func = compile_fx_inner(module, inputs, is_inference=is_inference, is_backward=is_backward)

    return compiled_func


def npu_compile(
    module: torch.nn.Module,
    inputs,
    *,
    is_inference: bool = False,
    is_backward: bool = False,
    **config_dict: Dict,
) -> torch.nn.Module:
    compiler = config_dict.get("compiler", "ge")
    if compiler == "ge":
        assert is_inference, "Currently, we use ge only for inference."
        return ge_compiler(module, inputs, **config_dict)
    elif compiler == "device_graph":
        assert is_inference, "Device graph capture/replay is intended for inference-style execution."
        return device_graph.device_graph_compiler(module, inputs, target="npu", **config_dict)
    else:
        return inductor_compiler(module, inputs, is_inference=is_inference, is_backward=is_backward, **config_dict)
