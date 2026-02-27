import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5)
    return output


def fn1(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5)
    return output, inputs_


def fn2(inputs, residual, weight, bias):
    residual = residual.to(torch.float32)
    weight = weight.to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
    inputs_ = inputs + residual
    output = torch.layer_norm(inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5)
    return output


def layernorm_test(xpu_graph, func, dynamic):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    residual = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024), device=device, dtype=data_type)
    bias = None

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    if func == fn0 or func == fn2:
        norm = compiled(inputs, residual, weight, bias)
        norm1 = func(inputs, residual, weight, bias)
        assert is_similar(norm1.cpu(), norm.cpu())
    if func == fn1:
        norm, res = compiled(inputs, residual, weight, bias)
        norm1, res1 = func(inputs, residual, weight, bias)
        assert is_similar(norm1.cpu(), norm.cpu())
        assert is_similar(res1.cpu(), res.cpu())


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=True, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [(fn0, False), (fn1, True), (fn2, False)],
    )
    def test_layernorm_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            layernorm_test(self.xpu_graph_backend, pattern_func, dynamic)
        # assert "Pattern.FusedAddLayerNorm changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)
    layernorm_test(xpu_graph_backend, fn0)
    layernorm_test(xpu_graph_backend, fn1)
    layernorm_test(xpu_graph_backend, fn2)
