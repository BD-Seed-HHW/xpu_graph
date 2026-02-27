
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


def naive_rmsnorm(inputs, weight):
    square = torch.square(inputs)
    mean = torch.mean(square, dim=-1, keepdim=True)
    root = torch.sqrt(mean + 1e-6)
    outputs = inputs / root * weight
    return outputs


def naive_rmsnorm_variant(inputs, weight):
    square = inputs**2
    mean = torch.mean(square, dim=-1, keepdim=True)
    iroot = torch.rsqrt(mean + 1e-6)
    outputs = weight * (inputs * iroot)
    return outputs


def naive_rmsnorm_noweight(inputs, weight):
    square = inputs * inputs
    mean = torch.mean(square, dim=-1, keepdim=True)
    root = (mean + 1e-3) ** 0.5
    normalized = inputs / root
    return normalized


def naive_rmsnorm_liftdtype(inputs, weight):
    return naive_rmsnorm(inputs.to(torch.float32), weight).to(inputs.dtype)


def naive_rmsnorm_liftparamdtype(inputs, weight):
    return naive_rmsnorm(inputs, weight.to(torch.float32)).to(inputs.dtype)


def rmsnorm_test(xpu_graph, func, input_dtype, weight_dtype, dynamic):
    inputs = torch.randn((8, 1024), device=device, dtype=input_dtype)
    if weight_dtype is not None:
        weight = torch.randn((1024,), device=device, dtype=weight_dtype)
    else:
        weight = None

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    norm = compiled(inputs, weight)
    norm1 = func(inputs, weight)
    assert is_similar(norm1, norm)


def rmsnorm_test_with_loss_and_grad(xpu_graph, func, input_dtype, weight_dtype, grad_dtype, dynamic):
    inputs = torch.randn((8, 1024), device=device, dtype=input_dtype, requires_grad=True)
    if weight_dtype is not None:
        weight = torch.randn((1024,), device=device, dtype=weight_dtype, requires_grad=True)
    else:
        weight = None
    dnorm = torch.randn((8, 1024), device=device, dtype=grad_dtype)

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)

    norm0 = compiled(inputs, weight)
    dinputs0, dweight0 = torch.autograd.grad((norm0,), (inputs, weight), (dnorm,), allow_unused=True)

    norm1 = func(inputs, weight)
    dinputs1, dweight1 = torch.autograd.grad((norm1,), (inputs, weight), (dnorm,), allow_unused=True)

    assert is_similar(norm0, norm1)
    assert is_similar(dinputs0, dinputs1)
    assert maybe_similar(dweight0, dweight1)


class TestRMSNorm:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func,input_dtype,weight_dtype",
        [
            (naive_rmsnorm, torch.float32, torch.float32),
            (naive_rmsnorm, torch.float16, torch.float32),
            (naive_rmsnorm_variant, torch.float16, torch.float16),
            (naive_rmsnorm_noweight, torch.float32, torch.float32),
            (naive_rmsnorm_liftdtype, torch.float32, torch.float16),
            (naive_rmsnorm_liftdtype, torch.float32, torch.bfloat16),
            (naive_rmsnorm_liftdtype, torch.float16, torch.float16),
            (naive_rmsnorm_liftdtype, torch.float16, torch.float32),
            (naive_rmsnorm_liftparamdtype, torch.float32, torch.float16),
            (naive_rmsnorm_liftparamdtype, torch.float16, torch.float16),
        ],
    )
    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    def test_rmsnorm_patterns(self, caplog, pattern_func, input_dtype, weight_dtype, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            rmsnorm_test(self.infer_backend, pattern_func, input_dtype, weight_dtype, dynamic)
        assert "Pattern.FusedRMSNorm changed graph" in caplog.text
        if pattern_func is naive_rmsnorm_liftdtype and input_dtype != torch.float32:
            assert "Pattern.RemoveRMSNormCast" in caplog.text
        if pattern_func is naive_rmsnorm_liftparamdtype and weight_dtype != torch.float32:
            assert "Pattern.RemoveRMSNormCast" in caplog.text

    @pytest.mark.parametrize(
        "pattern_func,input_dtype,weight_dtype,grad_dtype",
        [
            (naive_rmsnorm, torch.float32, torch.float32, torch.float32),
            (naive_rmsnorm_variant, torch.float16, torch.float32, torch.float32),
            (naive_rmsnorm_noweight, torch.float32, torch.float32, torch.float32),
            (naive_rmsnorm_liftdtype, torch.float32, torch.float16, torch.float32),
            (naive_rmsnorm_liftdtype, torch.float32, torch.bfloat16, torch.float32),
            (naive_rmsnorm_liftdtype, torch.float16, torch.float16, torch.float16),
            (naive_rmsnorm_liftdtype, torch.float16, torch.float32, torch.float32),
            (naive_rmsnorm_liftparamdtype, torch.float16, torch.float16, torch.float32),
            (naive_rmsnorm_liftparamdtype, torch.float32, torch.float16, torch.float32),
        ],
    )
    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    def test_rmsnorm_patterns_with_loss_and_grad(
        self, caplog, pattern_func, input_dtype, weight_dtype, grad_dtype, dynamic
    ):
        if self.train_backend._config.fallback_legacy_dispatch is False:
            pytest.skip("Pregrad passes will be replaced with joint passes")
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            rmsnorm_test_with_loss_and_grad(
                self.train_backend, pattern_func, input_dtype, weight_dtype, grad_dtype, dynamic
            )
        assert "Pattern.FusedRMSNorm changed graph" in caplog.text
        if pattern_func is naive_rmsnorm_liftdtype and input_dtype != torch.float32:
            assert "Pattern.RemoveRMSNormCast" in caplog.text
        if pattern_func is naive_rmsnorm_liftparamdtype and weight_dtype != torch.float32:
            assert "Pattern.RemoveRMSNormCast" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    rmsnorm_test(infer_backend, naive_rmsnorm, torch.float32, torch.float32)
    rmsnorm_test(infer_backend, naive_rmsnorm_variant, torch.float16, torch.float32)
    rmsnorm_test(infer_backend, naive_rmsnorm_noweight, torch.float32, torch.float32)
    rmsnorm_test(infer_backend, naive_rmsnorm_liftdtype, torch.float32, torch.float16)
    rmsnorm_test(infer_backend, naive_rmsnorm_liftparamdtype, torch.float32, torch.float16)
    rmsnorm_test(infer_backend, naive_rmsnorm_liftparamdtype, torch.float16, torch.float16)

    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    rmsnorm_test_with_loss_and_grad(train_backend, naive_rmsnorm, torch.float32, torch.float32)
    rmsnorm_test_with_loss_and_grad(train_backend, naive_rmsnorm_variant, torch.float16, torch.float32)
    rmsnorm_test_with_loss_and_grad(train_backend, naive_rmsnorm_noweight, torch.float32, torch.float32)
    rmsnorm_test_with_loss_and_grad(train_backend, naive_rmsnorm_liftdtype, torch.float32, torch.float16)
    rmsnorm_test_with_loss_and_grad(train_backend, naive_rmsnorm_liftparamdtype, torch.float32, torch.float16)
    rmsnorm_test_with_loss_and_grad(train_backend, naive_rmsnorm_liftparamdtype, torch.float16, torch.float16)
