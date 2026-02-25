import random

import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    aggregate_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, slice_, batch):
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_5)
    where_3 = torch.where(inputs, zeros, slice_6)
    output = torch.cat([slice_1, where_0, where_1, where_2, where_3], dim=-1)
    return output, slice_1


def fn1(inputs, slice_, batch):
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


def slice_where_cat_test(xpu_graph_backend, func, dynamic=True):
    batch = 512
    random_list = random.choices([0, 1], k=batch)
    inputs = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(batch, 35149, device=device, dtype=data_type)

    res = func(inputs, slice_, batch)

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
    res1 = compiled(inputs, slice_, batch)
    assert aggregate_similar(tuple(r.cpu().float() for r in res), tuple(r.cpu().float() for r in res1))


class TestSliceWhereCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            opt_level=OptLevel.level1,
            include_patterns=["CombinePointwiseSource", "CombinePointwiseSink"],
            is_training=False,
        )

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [
            (fn0, False),
            (fn1, True),
        ],
    )
    def test_slice_where_cat_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_where_cat_test(self.xpu_graph_backend, pattern_func, dynamic)
        assert "Pattern.CombinePointwiseSink changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2, is_training=False, debug=True)
    slice_where_cat_test(xpu_graph_backend, fn0)
    slice_where_cat_test(xpu_graph_backend, fn1)
