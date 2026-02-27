
import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    is_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
aten = torch.ops.aten


def naive_layer_norm(inputs, weight, bias):
    inputs_dtype = inputs.dtype
    inputs = inputs.to(torch.float32)
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    return outputs.to(inputs_dtype)


def complex_func(inputs, weight, bias):
    # dummy operations to generate a triton-kernel
    activated = inputs * torch.sigmoid(inputs)
    activated = activated * 0.1
    normalized = naive_layer_norm(activated, weight, bias)
    # dummy operations to generate a triton-kernel
    return normalized * normalized + 1e-6


def layernorm_test(xpu_graph, func, act_dtype, params_dtype):
    inputs = torch.randn((8, 1024), device=device, dtype=act_dtype)
    weight = torch.randn((1024,), device=device, dtype=params_dtype)
    bias = torch.randn((1024,), device=device, dtype=params_dtype)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    norm = compiled(inputs, weight, bias)
    norm1 = func(inputs, weight, bias)
    assert is_similar(norm1, norm)


def layernorm_test_with_loss_and_grad(xpu_graph, func, act_dtype, params_dtype, grad_dtype):
    inputs = torch.randn((8, 1024), device=device, dtype=act_dtype, requires_grad=True)
    weight = torch.randn((1024,), device=device, dtype=params_dtype, requires_grad=True)
    bias = torch.randn((1024,), device=device, dtype=params_dtype, requires_grad=True)
    dnorm = torch.randn((8, 1024), device=device, dtype=grad_dtype)

    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    norm0 = compiled(inputs, weight, bias)
    dinputs0, dweight0, dbias0 = torch.autograd.grad((norm0,), (inputs, weight, bias), (dnorm,), allow_unused=True)

    norm1 = func(inputs, weight, bias)
    dinputs1, dweight1, dbias1 = torch.autograd.grad((norm1,), (inputs, weight, bias), (dnorm,), allow_unused=True)

    assert is_similar(norm0, norm1)
    assert is_similar(dinputs0, dinputs1)
    assert is_similar(dweight0, dweight1)
    assert is_similar(dbias0, dbias1)


class TestLayerNorm:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "act_dtype,param_dtype",
        [
            (torch.float32, torch.float32),
            (torch.float32, torch.float16),
            (torch.float16, torch.float16),
            (torch.float32, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize(
        "pattern_func",
        [naive_layer_norm, complex_func],
    )
    def test_layernorm_patterns(self, caplog, pattern_func, act_dtype, param_dtype):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            layernorm_test(self.infer_backend, pattern_func, act_dtype, param_dtype)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text

    @pytest.mark.parametrize(
        "act_dtype,param_dtype,grad_dtype",
        [
            (torch.float32, torch.float32, torch.float32),
            (torch.float32, torch.float16, torch.float32),
            (torch.float32, torch.bfloat16, torch.float32),
        ],
    )
    @pytest.mark.parametrize(
        "pattern_func",
        [naive_layer_norm, complex_func],
    )
    def test_layernrom_patterns_with_loss_and_grad(self, caplog, pattern_func, act_dtype, param_dtype, grad_dtype):
        if self.train_backend._config.fallback_legacy_dispatch is False:
            pytest.skip("Pregrad passes will be replaced with joint passes")
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            layernorm_test_with_loss_and_grad(self.train_backend, pattern_func, act_dtype, param_dtype, grad_dtype)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    layernorm_test(infer_backend, complex_func, torch.float32, torch.float32)
    layernorm_test(infer_backend, complex_func, torch.float32, torch.float16)
    layernorm_test(infer_backend, complex_func, torch.float16, torch.float16)
    layernorm_test(infer_backend, complex_func, torch.float32, torch.bfloat16)

    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    layernorm_test_with_loss_and_grad(train_backend, complex_func, torch.float32, torch.float32, torch.float32)
    layernorm_test_with_loss_and_grad(train_backend, complex_func, torch.float32, torch.float16, torch.float32)
    layernorm_test_with_loss_and_grad(train_backend, complex_func, torch.float32, torch.bfloat16, torch.float32)
