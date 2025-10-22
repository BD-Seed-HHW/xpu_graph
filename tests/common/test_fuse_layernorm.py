
import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    is_similar,
    maybe_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def naive_layernorm(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    return outputs


def naive_layernorm_noweight(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) * torch.rsqrt(1e-6 + variance)
    outputs = bias + normalized
    return outputs


def naive_layernorm_noweight_nobias(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) / torch.sqrt(variance + 1e-5)
    return normalized


def naive_layernorm_nobias(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    return weight * normalized


def naive_layernorm_liftdtype(inputs, weight, bias):
    return naive_layernorm(inputs.to(torch.float32), weight, bias).to(inputs.dtype)


def naive_layernorm_liftparamdtype(inputs, weight, bias):
    return naive_layernorm(inputs, weight.to(torch.float32), bias.to(torch.float32)).to(inputs.dtype)


def layernorm_test(xpu_graph, func, input_dtype, weight_dtype, bias_dtype, dynamic):
    inputs = torch.randn((8, 1024), device=device, dtype=input_dtype)
    if weight_dtype is not None:
        weight = torch.randn((1024,), device=device, dtype=weight_dtype)
    else:
        weight = None
    if bias_dtype is not None:
        bias = torch.randn((1024,), device=device, dtype=bias_dtype)
    else:
        bias = None

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    norm = compiled(inputs, weight, bias)
    norm1 = func(inputs, weight, bias)
    assert is_similar(norm1, norm)


def layernorm_test_with_loss_and_grad(xpu_graph, func, input_dtype, weight_dtype, bias_dtype, grad_dtype, dynamic):
    inputs = torch.randn((8, 1024), device=device, dtype=input_dtype, requires_grad=True)
    if weight_dtype is not None:
        weight = torch.randn((1024,), device=device, dtype=weight_dtype, requires_grad=True)
    else:
        weight = None
    if bias_dtype is not None:
        bias = torch.randn((1024,), device=device, dtype=bias_dtype, requires_grad=True)
    else:
        bias = None
    dnorm = torch.randn((8, 1024), device=device, dtype=grad_dtype)

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)

    norm0 = compiled(inputs, weight, bias)
    dinputs0, dweight0, dbias0 = torch.autograd.grad((norm0,), (inputs, weight, bias), (dnorm,), allow_unused=True)

    norm1 = func(inputs, weight, bias)
    dinputs1, dweight1, dbias1 = torch.autograd.grad((norm1,), (inputs, weight, bias), (dnorm,), allow_unused=True)

    assert is_similar(norm0, norm1)
    assert is_similar(dinputs0, dinputs1)
    assert maybe_similar(dweight0, dweight1)
    assert maybe_similar(dbias0, dbias1)


class TestLayerNorm:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func,input_dtype,weight_dtype,bias_dtype",
        [
            (naive_layernorm, torch.float32, torch.float32, torch.float32),
            (naive_layernorm, torch.float16, torch.float32, torch.float32),
            (naive_layernorm_noweight, torch.float32, torch.float32, torch.float32),
            (naive_layernorm_noweight_nobias, torch.float32, torch.float32, None),
            (naive_layernorm_nobias, torch.float32, torch.float32, None),
            (naive_layernorm_liftdtype, torch.float32, torch.float16, torch.float16),
            (naive_layernorm_liftdtype, torch.float32, torch.bfloat16, torch.bfloat16),
            (naive_layernorm_liftdtype, torch.float16, torch.float16, torch.float16),
            (naive_layernorm_liftdtype, torch.float16, torch.float32, torch.float32),
            (naive_layernorm_liftparamdtype, torch.float32, torch.float16, torch.float16),
            (naive_layernorm_liftparamdtype, torch.float16, torch.float16, torch.float16),
        ],
    )
    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    def test_layernorm_patterns(self, caplog, pattern_func, input_dtype, weight_dtype, bias_dtype, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            layernorm_test(self.infer_backend, pattern_func, input_dtype, weight_dtype, bias_dtype, dynamic)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text
        if pattern_func is naive_layernorm_liftdtype and input_dtype != torch.float32:
            assert "Pattern.RemoveLayerNormCast" in caplog.text
        if pattern_func is naive_layernorm_liftparamdtype and (
            weight_dtype != torch.float32 or bias_dtype != torch.float32
        ):
            assert "Pattern.RemoveLayerNormCast" in caplog.text

    @pytest.mark.parametrize(
        "pattern_func,input_dtype,weight_dtype,bias_dtype,grad_dtype",
        [
            (naive_layernorm, torch.float32, torch.float32, torch.float32, torch.float32),
            (naive_layernorm_noweight, torch.float32, torch.float32, torch.float32, torch.float32),
            (naive_layernorm_noweight_nobias, torch.float32, torch.float32, torch.float32, torch.float32),
            (naive_layernorm_nobias, torch.float32, torch.float32, torch.float32, torch.float32),
            (naive_layernorm_liftdtype, torch.float32, torch.float16, torch.float16, torch.float32),
            (naive_layernorm_liftdtype, torch.float32, torch.bfloat16, torch.bfloat16, torch.float32),
            (naive_layernorm_liftdtype, torch.float16, torch.float16, torch.float16, torch.float16),
            (naive_layernorm_liftdtype, torch.float16, torch.float32, torch.float32, torch.float32),
            (naive_layernorm_liftparamdtype, torch.float16, torch.float16, torch.float16, torch.float32),
            (naive_layernorm_liftparamdtype, torch.float32, torch.float16, torch.float16, torch.float32),
        ],
    )
    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    def test_layernrom_patterns_with_loss_and_grad(
        self, caplog, pattern_func, input_dtype, weight_dtype, bias_dtype, grad_dtype, dynamic
    ):
        if self.train_backend._config.fallback_legacy_dispatch is False:
            pytest.skip("Pregrad passes will be replaced with joint passes")
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            layernorm_test_with_loss_and_grad(
                self.train_backend, pattern_func, input_dtype, weight_dtype, bias_dtype, grad_dtype, dynamic
            )
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text
        if pattern_func is naive_layernorm_liftdtype and input_dtype != torch.float32:
            assert "Pattern.RemoveLayerNormCast" in caplog.text
        if pattern_func is naive_layernorm_liftparamdtype and (
            weight_dtype != torch.float32 or bias_dtype != torch.float32
        ):
            assert "Pattern.RemoveLayerNormCast" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    layernorm_test(infer_backend, naive_layernorm, torch.float32, torch.float32, torch.float32)
    layernorm_test(infer_backend, naive_layernorm_noweight, torch.float32, torch.float32, torch.float32)
    layernorm_test(infer_backend, naive_layernorm_noweight_nobias, torch.float32, torch.float32, None)
    layernorm_test(infer_backend, naive_layernorm_nobias, torch.float32, torch.float32, None)
    layernorm_test(infer_backend, naive_layernorm_liftdtype, torch.float32, torch.float16, torch.float16)
    layernorm_test(infer_backend, naive_layernorm_liftdtype, torch.float32, torch.bfloat16, torch.bfloat16)
    layernorm_test(infer_backend, naive_layernorm_liftdtype, torch.float16, torch.float16, torch.float16)
    layernorm_test(infer_backend, naive_layernorm_liftdtype, torch.float16, torch.float32, torch.float32)

    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm, torch.float32, torch.float32, torch.float32, torch.float32
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_noweight, torch.float32, torch.float32, torch.float32, torch.float32
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_noweight_nobias, torch.float32, torch.float32, None, torch.float32
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_nobias, torch.float32, torch.float32, None, torch.float32
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_liftdtype, torch.float32, torch.float16, torch.float16, torch.float32
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_liftdtype, torch.float32, torch.bfloat16, torch.bfloat16, torch.float32
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_liftdtype, torch.float16, torch.float16, torch.float16, torch.float16
    )
    layernorm_test_with_loss_and_grad(
        train_backend, naive_layernorm_liftdtype, torch.float16, torch.float32, torch.float32, torch.float32
    )
