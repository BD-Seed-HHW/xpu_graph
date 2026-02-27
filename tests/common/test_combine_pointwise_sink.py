
import pytest
import torch
from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    aggregate_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0_concat_varlen(inputs, slice_):
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10182]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11096]
    slice_6 = slice_[:, 11445:11477]
    where_0 = inputs * slice_2
    where_1 = inputs * slice_4
    where_2 = inputs * slice_5
    where_3 = inputs * slice_6
    output = torch.concat([slice_1, where_0, where_1, where_2, where_3], dim=-1)
    return output, slice_1


def fn0_output(inputs, slice_):
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = inputs * slice_2
    where_1 = inputs * slice_4
    where_2 = inputs * slice_5
    where_3 = inputs * slice_6
    return slice_1, where_0, where_1, where_2, where_3


def fn1(inputs, slice_):
    slice_1 = slice_[:, 0:4]
    slice_2 = slice_[:, 11984:11988]
    slice_3 = slice_[:, 11562:11566]
    slice_4 = slice_[:, 9782:9786]
    slice_5 = slice_[:, 9727:9731]
    slice_6 = slice_[:, 9745:9749]
    slice_7 = slice_[:, 9763:9767]
    slice_8 = slice_[:, 9773:9777]
    slice_9 = slice_[:, 9805:9809]
    slice_10 = slice_[:, 9822:9826]
    slice_11 = slice_[:, 10283:10287]
    slice_12 = slice_[:, 10736:10740]
    where_0 = slice_2 + inputs
    where_1 = slice_3 + inputs
    where_2 = slice_4 + inputs
    where_3 = slice_5 + inputs
    where_4 = slice_6 + inputs
    where_5 = slice_7 + inputs
    where_6 = slice_8 + inputs
    where_7 = slice_9 + inputs
    where_8 = slice_10 + inputs
    where_9 = slice_11 + inputs
    where_10 = slice_12 + inputs
    stack = torch.stack(
        [
            slice_1,
            where_0,
            where_1,
            where_2,
            where_3,
            where_4,
            where_5,
            where_6,
            where_7,
            where_8,
            where_9,
            where_10,
        ],
        dim=0,
    )
    return stack, slice_1


def fn2_concat_varlen(inputs, slice_):
    batch = inputs.size(0)
    inputs = inputs > 0
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    zeros2 = torch.zeros([batch, 64], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118 : (10118 + 64)]
    slice_4 = slice_[:, 10579 : (10579 + 32)]
    slice_5 = slice_[:, 11032 : (11032 + 64)]
    slice_6 = slice_[:, 11445 : (11445 + 32)]
    slice_7 = slice_[:, 11477 : (11477 + 32)]
    slice_8 = slice_[:, 11519 : (11519 + 64)]
    where_0 = torch.where(inputs, zeros2, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros2, slice_5)
    where_3 = torch.where(inputs, zeros, slice_6)
    where_4 = torch.where(inputs, zeros, slice_7)
    where_5 = torch.where(inputs, zeros2, slice_8)
    output = torch.cat([slice_1, where_0, where_1, where_2, where_3, where_4, where_5], dim=-1)
    return output, slice_1


def fn3(inputs, slice_):
    batch = inputs.size(0)
    inputs = inputs > 0
    zeros = torch.zeros([batch, 4], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:4]
    slice_2 = slice_[:, 11984:11988]
    slice_3 = slice_[:, 11562:11566]
    slice_4 = slice_[:, 9782:9786]
    slice_5 = slice_[:, 9727:9731]
    slice_6 = slice_[:, 9745:9749]
    slice_7 = slice_[:, 9763:9767]
    slice_8 = slice_[:, 9773:9777]
    slice_9 = slice_[:, 9805:9809]
    slice_10 = slice_[:, 9822:9826]
    slice_11 = slice_[:, 10283:10287]
    slice_12 = slice_[:, 10736:10740]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_3)
    where_2 = torch.where(inputs, zeros, slice_4)
    where_3 = torch.where(inputs, zeros, slice_5)
    where_4 = torch.where(inputs, zeros, slice_6)
    where_5 = torch.where(inputs, zeros, slice_7)
    where_6 = torch.where(inputs, zeros, slice_8)
    where_7 = torch.where(inputs, zeros, slice_9)
    where_8 = torch.where(inputs, zeros, slice_10)
    where_9 = torch.where(inputs, zeros, slice_11)
    where_10 = torch.where(inputs, zeros, slice_12)
    stack = torch.stack(
        [
            slice_1,
            where_0,
            where_1,
            where_2,
            where_3,
            where_4,
            where_5,
            where_6,
            where_7,
            where_8,
            where_9,
            where_10,
        ],
        dim=0,
    )
    return stack, slice_1


def fn4(inputs, slice_):
    inputs = inputs > 0
    batch = inputs.size(0)
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    ones = torch.ones([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_5)
    where_3 = torch.where(inputs, zeros, slice_6)
    where_4 = torch.where(inputs, ones, slice_2)
    where_5 = torch.where(inputs, ones, slice_4)
    where_6 = torch.where(inputs, ones, slice_5)
    where_7 = torch.where(inputs, ones, slice_6)
    output = torch.cat([slice_1, where_0, where_1, where_2, where_3, where_4, where_5, where_6, where_7], dim=-1)
    output_1 = torch.stack([slice_1, where_0, where_1, where_2, where_3, where_4, where_5, where_6, where_7]).sum(dim=0)
    add_0 = where_0 * where_4
    add_1 = where_1 * where_5
    add_2 = where_2 * where_6
    add_3 = where_3 * where_7
    return output, output_1, add_0, add_1, add_2, add_3


def fn4_xfail(inputs, slice_):
    inputs = inputs > 0
    batch = inputs.size(0)
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    ones = torch.ones([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_5)
    where_3 = torch.where(inputs, zeros, slice_6)
    where_4 = torch.where(inputs, ones, slice_2)
    where_5 = torch.where(inputs, ones, slice_4)
    where_6 = torch.where(inputs, ones, slice_5)
    where_7 = torch.where(inputs, ones, slice_6)
    add_0 = where_0 * where_4
    add_1 = where_1 * where_5
    add_2 = where_2 * where_6
    add_3 = where_3 * where_7
    output = torch.cat(
        [add_0, add_1, add_2, add_3, slice_1, where_0, where_1, where_2, where_3, where_4, where_5, where_6, where_7],
        dim=-1,
    )
    output_1 = torch.stack(
        [add_0, add_1, add_2, add_3, slice_1, where_0, where_1, where_2, where_3, where_4, where_5, where_6, where_7]
    ).sum(dim=0)
    return output, output_1


def fn5_symshape(inputs, slice_):
    seq_len = inputs.size(0)
    l0, l1, l2, _l3 = seq_len // 4, (seq_len + 1) // 4, (seq_len + 2) // 4, (seq_len + 3) // 4
    o0, o1, o2, o3 = 0, l0, l0 + l1, l0 + l1 + l2
    a0 = slice_[o0:o1]
    a1 = slice_[o1:o2]
    a2 = slice_[o2:o3]
    a3 = slice_[o3:]

    res = torch.cat([a0 * 0.1, a1 * 0.1, a2 * 0.1, a3], dim=0).sum(dim=0)
    return res


def combine_pointwise_test(xpu_graph_backend, func, is_training=False):
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)

    for batch in [8, 10, 80]:
        inputs = torch.randn(batch, device=device, dtype=data_type).unsqueeze(-1)
        slice_ = torch.randn(batch, 35149, device=device, dtype=data_type)
        if is_training:
            inputs = inputs.requires_grad_()

        res = func(inputs, slice_)
        res1 = compiled(inputs, slice_)
        assert aggregate_similar(res, res1, atol=1e-5, rtol=1e-5)


class TestCombinePointwiseSinkInference:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                opt_level=OptLevel.level2,
            )
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0_concat_varlen,
            fn0_output,
            fn1,
            fn2_concat_varlen,
            fn3,
            fn4,
            fn4_xfail,
            fn5_symshape,
        ],
    )
    def test_pointwise_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_pointwise_test(self.xpu_graph_backend, pattern_func)
        if "xfail" in pattern_func.__name__:
            assert "Pattern.CombinePointwiseSink changed graph" not in caplog.text
        else:
            assert "Pattern.CombinePointwiseSink changed graph" in caplog.text
        if "concat_varlen" in pattern_func.__name__:
            assert "aten.stack.default" not in caplog.text


class TestCombinePointwiseSinkTraining:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=True,
                opt_level=OptLevel.level2,
            )
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0_concat_varlen,
            fn0_output,
            fn1,
            fn5_symshape,
        ],
    )
    def test_pointwise_patterns(self, caplog, pattern_func):
        if self.xpu_graph_backend._config.fallback_legacy_dispatch is False:
            pytest.skip("Pregrad passes will be replaced with joint passes")
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_pointwise_test(self.xpu_graph_backend, pattern_func, is_training=True)
        if "xfail" in pattern_func.__name__:
            assert "Pattern.CombinePointwiseSink changed graph" not in caplog.text
        else:
            assert caplog.text.count("Pattern.CombinePointwiseSink changed graph") == 2
        if "concat_varlen" in pattern_func.__name__:
            assert "aten.stack.default" not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True))
    combine_pointwise_test(xpu_graph_backend, fn0_concat_varlen)
    combine_pointwise_test(xpu_graph_backend, fn1)
    combine_pointwise_test(xpu_graph_backend, fn2_concat_varlen)
    combine_pointwise_test(xpu_graph_backend, fn4)
