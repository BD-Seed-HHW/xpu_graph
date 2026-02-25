import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache


class LayerNorm1(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, keepdim=True, correction=False)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype) + self.bias

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LayerNorm2(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, keepdim=True, correction=False)
        hidden_states = (hidden_states - mean) / torch.sqrt(self.variance_epsilon + variance)
        return self.bias + hidden_states.to(input_dtype) * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def layernorm_test(xpu_graph, ModCls, dynamic, input_dtype, param_dtype):
    mod = ModCls(10).mlu().to(param_dtype)
    mod_compiled = ModCls(10).mlu().to(param_dtype)
    mod_compiled.load_state_dict(mod.state_dict())
    with torch.inference_mode():
        a = torch.randn(2, 10).to(input_dtype).mlu()

        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=xpu_graph, dynamic=dynamic)
        norm = mod_compiled.forward(a)
        norm1 = mod.forward(a)
    assert is_similar(norm1.cpu().float(), norm.cpu().float())


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_mod,dynamic,input_dtype,param_dtype",
        [
            (LayerNorm1, True, torch.float32, torch.float32),
            (LayerNorm2, True, torch.float16, torch.float16),
            (LayerNorm1, False, torch.float32, torch.float16),
            (LayerNorm2, False, torch.float16, torch.float32),
            (LayerNorm1, True, torch.float32, torch.bfloat16),
        ],
    )
    def test_layernorm_patterns(self, caplog, pattern_mod, dynamic, input_dtype, param_dtype):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            layernorm_test(self.xpu_graph_backend, pattern_mod, dynamic, input_dtype, param_dtype)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text
        if input_dtype != torch.float32:
            assert "Pattern.RemoveLayerNormCast" in caplog.text
        assert "Pattern.CustomLayerNorm changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, debug=True)
    layernorm_test(xpu_graph_backend, LayerNorm1, torch.float32, torch.float32)
    # layernorm_test(xpu_graph_backend, LayerNorm2, torch.float16, torch.float16)
    # layernorm_test(xpu_graph_backend, LayerNorm1, torch.float32, torch.float16)
    # layernorm_test(xpu_graph_backend, LayerNorm2, torch.float16, torch.float32)
    # layernorm_test(xpu_graph_backend, LayerNorm1, torch.float32, torch.bfloat16)
