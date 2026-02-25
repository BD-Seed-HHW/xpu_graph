import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar


def naive(q, k, v, bias, causal, sm_scale, has_bias):
    N_CTX = q.shape[-2]
    n_head = q.shape[1]
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="mlu"))
    p = torch.matmul(q, k.transpose(2, 3)).float() * sm_scale
    if has_bias:
        bias = bias.unsqueeze(1).repeat(1, n_head, 1, 1)
        p = p.masked_fill(bias == 0, 0.0)
    if causal:  # false
        p[:, :, M == 0] = float("-1e6")
    p = torch.nn.functional.silu(p).to(q.dtype)
    ref_out = torch.matmul(p, v)
    return ref_out


def fn0(q, k, v, bias):
    causal = False
    sm_scale = 1.0
    has_bias = True
    return naive(q, k, v, bias, causal, sm_scale, has_bias)


def fn1(q, k, v, bias):
    causal = False
    sm_scale = 1.0
    has_bias = False
    return naive(q, k, v, None, causal, sm_scale, has_bias)


def linear_attention_test(xpu_graph, func, dynamic=True):
    dtype = torch.float16
    device = "mlu"
    BATCH, H, N_CTX, D_HEAD = 32, 8, 1024, 128
    with torch.no_grad():
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        bias = torch.randn((BATCH, N_CTX, N_CTX), dtype=dtype, device=device)

        compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
        res = compiled(q, k, v, bias)
        res1 = func(q, k, v, bias)
    is_similar(res.cpu().float(), res1.cpu().float())


class TestLinearAttention:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        config.target = xpu_graph.config.Target.mlu
        config.vendor_compiler = {"mode": "reduce-overhead"}
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [
            (fn0, True),
            (fn1, False),
        ],
    )
    def test_linear_attention_patterns(self, pattern_func, dynamic):
        linear_attention_test(self.xpu_graph, pattern_func, dynamic)


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    config.target = xpu_graph.config.Target.mlu
    config.opt_level = OptLevel.level2
    xpu_graph_backend = xpu_graph.compiler.XpuGraph(config)
    linear_attention_test(xpu_graph_backend, fn0)
    linear_attention_test(xpu_graph_backend, fn1)
