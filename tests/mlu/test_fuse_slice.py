import pytest
import torch
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
aten = torch.ops.aten


def fn12(x):
    x1 = x[:, 34691:34755] + 1
    x2 = x[:, 38371:38435] + 1
    x3 = x[:, 41479:41543] + 1
    x4 = x[:, 33535:33599] + 1
    x5 = x[:, 34065:34129] + 1
    x6 = x[:, 39415:39479] + 1
    x7 = x[:, 40264:40328] + 1
    x8 = x[:, 24347:24411] + 1
    x9 = x[:, 23854:23918] + 1
    x10 = x[:, 23361:23425] + 1
    x11 = x[:, 34691:34755] + 1
    x12 = x[:, 38371:38435] + 1
    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12


def fn13(x):
    x1 = x[:, 0:64] + 1
    x2 = x[:, 64:128] + 1
    x3 = x[:, 128:192] + 1
    x4 = x[:, 192:256] + 1
    x5 = x[:, 256:384] + 1
    x6 = x[:, 384:512] + 1
    x7 = x[:, 512:640] + 1
    return x1, x2, x3, x4, x5, x6, x7


def fn14(x):
    tmp_Test = [
        12573,
        11984,
        11562,
        9782,
        9727,
        9745,
        9763,
        9773,
        9805,
        9822,
        10283,
        10736,
    ]
    input_list = []
    for index in tmp_Test:
        input_list.append(x[:, index : index + 4] + 1)
    return input_list


def fn15(x):
    tmp_Test = [
        9425,
        18012,
        11675,
        12033,
        9926,
        10387,
        10840,
        11253,
        9457,
        18044,
        11707,
        12065,
        9958,
        10419,
        10872,
        11285,
        9489,
        18076,
        11739,
        12097,
        9990,
        10451,
        10904,
        11317,
        9521,
        18108,
        11771,
        12129,
        10022,
        10483,
        10936,
        11349,
        9553,
        18140,
        11803,
        12161,
        10054,
        10515,
        10968,
        11381,
        9585,
        18172,
        11835,
        12193,
        10086,
        10547,
        11000,
        11413,
        9617,
        18204,
        11867,
        12225,
        10118,
        10579,
        11032,
        11445,
    ]
    input_list = []
    for index in tmp_Test:
        input_list.append(x[:, index : index + 32] + 1)
    return input_list


def fn16(x):
    tmp_Test = [
        12573,
        11984,
        11562,
        9782,
        9727,
        9745,
        9763,
        9773,
        9805,
        9822,
        10283,
        10736,
    ]
    input_list = []
    for index in tmp_Test:
        input_list.append(x[:, :, index : index + 4] + 1)
    return input_list


def slice_test(xpu_graph_backend, func, dynamic=True):
    for batch in (10, 512, 64, 31):
        if func in [fn16]:
            a = torch.randn(batch, 12, 43106).to(device=device)
        else:
            a = torch.randn(batch, 43106).to(device=device)

        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
        res = compiled(a)[0]
        res1 = func(a)[0]
        assert torch.equal(res.cpu().float(), res1.cpu().float())


class TestSlice:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level1, vendor_compiler_config=None
        )

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [
            (fn12, False),
            (fn13, True),
            (fn14, False),
            (fn15, True),
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_test(self.xpu_graph_backend, pattern_func, dynamic)
        assert "Pattern.FusedSlice changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=True, opt_level=OptLevel.level1, debug=False)
    slice_test(xpu_graph_backend, fn12)
    slice_test(xpu_graph_backend, fn16)
