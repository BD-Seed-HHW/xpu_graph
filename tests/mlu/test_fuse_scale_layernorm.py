# ruff: noqa: F841
import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

torch._inductor.config.comprehensive_padding = False

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, weight, bias, scale, q_weight, q_bias):
    inputs = inputs * scale
    origin_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=False)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=origin_dtype)
    return outputs


def layernorm_test(xpu_graph, func, dynamic):
    inputs = torch.randn((64, 1362), device=device, dtype=data_type)
    residual = torch.randn((64, 1362), device=device, dtype=data_type)
    weight = torch.randn((1362), device=device, dtype=data_type)
    bias = torch.randn((1362,), device=device, dtype=data_type)
    q_weight = torch.randn((1362, 1362), device=device, dtype=data_type)
    scale = 0.1

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)

    q_bias = None
    norm = compiled(inputs, weight, bias, scale, q_weight, q_bias)
    norm1 = func(inputs, weight, bias, scale, q_weight, q_bias)
    is_similar(norm1.cpu().float(), norm.cpu().float())


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=True, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    @pytest.mark.parametrize("dynamic", [True, False])
    def test_layernorm_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            layernorm_test(self.xpu_graph_backend, pattern_func, dynamic)
        assert "Pattern.FusedScaleLayernorm changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=True, opt_level=OptLevel.level2, debug=True)
    layernorm_test(xpu_graph_backend, fn0)
