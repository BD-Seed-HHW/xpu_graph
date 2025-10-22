
import pytest
import torch
from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0(a):
    a_list = a.split(8, dim=-1)
    res = 0
    for a in a_list:
        res += (a + 0.1).sum(dim=-1)
    return res


def fn0_split_varlen(a):
    a_list = a.split(10, dim=-1)
    res = 0
    for a in a_list:
        res += (a + 0.2).sum(dim=-1)
    return res


def fn0_split_varlen2(a):
    a_list = a.split([6, 10, 20, a.size(-1) - 36], dim=-1)
    res = 0
    for a in a_list:
        res += (a + 0.2).sum(dim=-1)
    return res


def fn1(a):
    a = torch.reshape(a, [a.size(0), 16, 4])
    a_list = a.unbind(dim=-1)
    res = 0
    for a in a_list:
        res += a * 0.1
    return res


def fn2(a):
    a = torch.reshape(a, [a.size(0), 16, 4])
    a_list = a.unbind(dim=-1)
    res = 0
    for a in a_list[:-1]:
        res += a * 0.1
    res += a_list[-1]
    return res


def fn2_extra_inputs(a, b, c, d):
    a = torch.reshape(a, [a.size(0), 16, 4])
    a_list = a.unbind(dim=-1)
    res = 0
    for a in a_list[:-1]:
        res += a + b
    res += c + d
    res += a_list[-1]
    return res


def fn3_inputs(x1, x2, x3, x4):
    return x1 * 0.1 + x2 * 0.1 + x3 * 0.1 + x4


def fn4_symshape(a):
    a0, a1, a2, a3 = a.split([a.size(0) // 4, (a.size(0) + 1) // 4, (a.size(0) + 2) // 4, (a.size(0) + 3) // 4], dim=0)
    res = (a0 * 0.1).sum(dim=0) + (a1 * 0.1).sum(dim=0) + (a2 * 0.1).sum(dim=0) + a3.sum(dim=0)
    return res


def fn5(a):
    a_list = a.split(8, dim=-1)
    res = a_list[0].sum(dim=-1)
    for a in a_list:
        res += (a + 0.1).sum(dim=-1)
    return res


def combine_pointwise_test(xpu_graph_backend, func, is_training=False):
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)

    for batch in [8, 10, 80]:
        inputs = torch.randn(batch, 64, device=device, dtype=data_type)
        if is_training:
            inputs = inputs.requires_grad_()

        if func == fn3_inputs:
            inputs = torch.reshape(inputs, [batch, 16, 4]).unbind(dim=-1)
        elif func == fn2_extra_inputs:
            b = torch.ones((batch, 16), device=device, dtype=data_type)
            c = torch.ones((batch, 16), device=device, dtype=data_type)
            d = torch.ones((batch, 16), device=device, dtype=data_type)
            inputs = (inputs, b, c, d)
        else:
            inputs = (inputs,)

        res = func(*inputs)
        res1 = compiled(*inputs)
        assert is_similar(res, res1, atol=1e-5, rtol=1e-5)


class TestCombinePointwiseSourceInference:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                opt_level=OptLevel.level1,
                include_patterns=["CombinePointwiseSource"],
            )
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn0_split_varlen,
            fn0_split_varlen2,
            fn1,
            fn2,
            fn2_extra_inputs,
            fn3_inputs,
            fn4_symshape,
            fn5,
        ],
    )
    def test_pointwise_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_pointwise_test(self.xpu_graph_backend, pattern_func)
        if "xfail" in pattern_func.__name__:
            assert "Pattern.CombinePointwiseSource changed graph" not in caplog.text
        else:
            assert caplog.text.count("Pattern.CombinePointwiseSource changed graph") == 2
        if "split_varlen" in pattern_func.__name__:
            assert "aten.stack.default" not in caplog.text


class TestCombinePointwiseSourceTraining:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=True,
                opt_level=OptLevel.level1,
                include_patterns=["CombinePointwiseSource"],
            )
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn0_split_varlen,
            fn0_split_varlen2,
            fn1,
            fn2,
            fn2_extra_inputs,
            fn3_inputs,
            fn4_symshape,
            fn5,
        ],
    )
    def test_pointwise_patterns(self, caplog, pattern_func):
        if self.xpu_graph_backend._config.fallback_legacy_dispatch is False:
            pytest.skip("Pregrad passes will be replaced with joint passes")
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_pointwise_test(self.xpu_graph_backend, pattern_func, is_training=True)
        if "xfail" in pattern_func.__name__:
            assert "Pattern.CombinePointwiseSource changed graph" not in caplog.text
        else:
            assert caplog.text.count("Pattern.CombinePointwiseSource changed graph") == 2
        if "split_varlen" in pattern_func.__name__:
            assert "aten.stack.default" not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True))
    # combine_pointwise_test(xpu_graph_backend, fn0)
    # combine_pointwise_test(xpu_graph_backend, fn1)
    # combine_pointwise_test(xpu_graph_backend, fn2)
    combine_pointwise_test(xpu_graph_backend, fn5)
