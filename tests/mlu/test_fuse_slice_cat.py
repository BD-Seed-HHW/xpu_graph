import pytest
import torch
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
aten = torch.ops.aten


def fn0(x):
    axis = -1
    c = torch.cat(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        axis,
    )
    return (c,)


def fn1(x):
    axis = -1
    c = torch.cat(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        axis,
    )
    c = c + 1
    slice_190 = x[:, 35263:35295]
    slice_192 = x[:, 35296:35328]
    slice_194 = x[:, 35329:35361]
    slice_196 = x[:, 35378:35410]
    slice_198 = x[:, 35411:35443]
    slice_200 = x[:, 35444:35476]
    slice_202 = x[:, 35971:36003]
    slice_204 = x[:, 37206:37238]
    slice_206 = x[:, 37256:37288]
    slice_208 = x[:, 37306:37338]
    slice_210 = x[:, 37391:37423]
    slice_212 = x[:, 37432:37464]
    slice_214 = x[:, 37474:37506]
    slice_216 = x[:, 1165:1197]
    slice_218 = x[:, 1340:1372]
    slice_220 = x[:, 1387:1419]
    slice_222 = x[:, 1739:1771]
    slice_224 = x[:, 2629:2661]
    slice_226 = x[:, 2676:2708]
    slice_228 = x[:, 2723:2755]
    slice_230 = x[:, 2897:2929]
    slice_232 = x[:, 2186:2218]
    slice_234 = x[:, 2232:2264]
    slice_236 = x[:, 2278:2310]
    slice_238 = x[:, 2417:2449]

    stack_4 = torch.cat(
        [
            slice_190,
            slice_192,
            slice_194,
            slice_196,
            slice_198,
            slice_200,
            slice_202,
            slice_204,
            slice_206,
            slice_208,
            slice_210,
            slice_212,
            slice_214,
            slice_216,
            slice_218,
            slice_220,
            slice_222,
            slice_224,
            slice_226,
            slice_228,
            slice_230,
            slice_232,
            slice_234,
            slice_236,
            slice_238,
        ],
        dim=-1,
    )
    return (torch.cat([stack_4, c], dim=-1),)


def fn2(arg41_1):
    slice_190 = arg41_1[:, 35263:35295]
    slice_192 = arg41_1[:, 35296:35328]
    slice_194 = arg41_1[:, 35329:35361]
    slice_196 = arg41_1[:, 35378:35410]
    slice_198 = arg41_1[:, 35411:35443]
    slice_200 = arg41_1[:, 35444:35476]
    slice_202 = arg41_1[:, 35971:36003]
    slice_204 = arg41_1[:, 37206:37238]
    slice_206 = arg41_1[:, 37256:37288]
    slice_208 = arg41_1[:, 37306:37338]
    slice_210 = arg41_1[:, 37391:37423]
    slice_212 = arg41_1[:, 37432:37464]
    slice_214 = arg41_1[:, 37474:37506]
    slice_216 = arg41_1[:, 1165:1197]
    slice_218 = arg41_1[:, 1340:1372]
    slice_220 = arg41_1[:, 1387:1419]
    slice_222 = arg41_1[:, 1739:1771]
    slice_224 = arg41_1[:, 2629:2661]
    slice_226 = arg41_1[:, 2676:2708]
    slice_228 = arg41_1[:, 2723:2755]
    slice_230 = arg41_1[:, 2897:2929]
    slice_232 = arg41_1[:, 2186:2218]
    slice_234 = arg41_1[:, 2232:2264]
    slice_236 = arg41_1[:, 2278:2310]
    slice_238 = arg41_1[:, 2417:2449]

    stack_4 = torch.stack(
        [
            slice_190,
            slice_192,
            slice_194,
            slice_196,
            slice_198,
            slice_200,
            slice_202,
            slice_204,
            slice_206,
            slice_208,
            slice_210,
            slice_212,
            slice_214,
            slice_216,
            slice_218,
            slice_220,
            slice_222,
            slice_224,
            slice_226,
            slice_228,
            slice_230,
            slice_232,
            slice_234,
            slice_236,
            slice_238,
        ],
        dim=0,
    )

    sum_6 = torch.sum(stack_4, dim=0)
    return (sum_6,)


def fn3(x):
    axis = -1
    c = torch.cat(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        axis,
    )
    c = c + 1
    d = torch.stack(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        0,
    )
    return (c, d)


def fn5(x):
    axis = -1
    b = x.clone()
    c = torch.cat(
        [
            b,
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
        ],
        axis,
    )
    return (c, 1)


def fn6(x):
    b = x.clone()
    d = b.clone()
    c = aten.cat(
        [
            b,
            x[:, 34691:34755],
            d[:, 38371:38435],
            d[:, 38371:38437],
        ],
        1,
    )
    return (c, 1)


def fn7(x):
    b = x.clone()
    d = b.clone()
    c = aten.cat(
        [
            b,
            x[:, 34691:34755],
            x[:, 38371:38435],
            d,
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
        ],
        1,
    )
    return (c, 1)


def fn8(x):
    b = x[:, 39415:39479].contiguous()
    b += 1
    c = aten.stack(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 34065:34129],
            b,
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
    )
    return (c, 1)


def fn9(x):
    c = aten.cat(
        [
            x[:, 43106 - 2 : 9223372036854775807],
            x[:, 43106 - 2 : 9223372036854775807],
        ],
        1,
    )
    return (c, 1)


def fn10(x):
    c = aten.cat(
        [
            x[:, 43106 - 2 :],
            x[:, 43106 - 2 :],
        ],
        1,
    )
    return (c, 1)


def fn11(x):
    c = aten.cat(
        [
            x[:, 43106 - 2 : -1],
            x[:, 43106 - 2 : -1],
        ],
        1,
    )
    return (c, 1)


def slice_test(xpu_graph_backend, func, dynamic=True):
    for batch in (10, 512, 256, 128, 64, 32):
        a = torch.randn(batch, 43106).to(device=device)

        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
        res = compiled(a)[0]
        res1 = func(a)[0]
        assert torch.equal(res.cpu().float(), res1.cpu().float())


class TestCatSlice:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            freeze=True,
            opt_level=OptLevel.level1,
            vendor_compiler_config=None,
        )

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [
            (fn0, True),
            (fn1, False),
            (fn5, True),
            (fn6, False),
            (fn7, True),
            (fn9, False),
            (fn10, True),
            (fn11, False),
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_test(self.xpu_graph_backend, pattern_func, dynamic)
        assert "Pattern.FusedCatSlice changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=True, opt_level=OptLevel.level1, debug=False, vendor_compiler_config=None
    )
    slice_test(xpu_graph_backend, fn7)
