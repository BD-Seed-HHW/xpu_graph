import pytest
import torch
import torch.nn.functional as F
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    return output + bias if bias is not None else output


def fn1(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output + bias
    return output


def fn2(inputs, weight, residual):
    output = torch.matmul(inputs, weight)
    output = output + residual
    return output


def fn3(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output
    output = F.gelu(output)
    return output


def fn4(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output + bias
    output = F.gelu(output)
    return output


def fn5(inputs, weight, residual):
    output = torch.matmul(inputs, weight)
    output = output + residual
    output = F.gelu(output)
    return output


def fn6(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    return output.view(1, 5, 32, 256)


def fn7(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    return output.view(5, 32, 256)


def fn8(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output + bias
    return output.view(1, 5, 32, 256)


def fn9(inputs, weight, residual):
    output = torch.matmul(inputs, weight)
    output = output + residual
    return output.view(1, 5, 32, 256)


def fn10(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output
    output = F.gelu(output)
    return output.view(1, 5, 32, 256)


def fn11(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output + bias
    output = F.gelu(output)
    return output.view(1, 5, 32, 256)


def fn12(inputs, weight, residual):
    output = torch.matmul(inputs, weight)
    output = output + residual
    output = F.gelu(output)
    return output.view(1, 5, 32, 256)


def bmm_test(xpu_graph_backend, func, dynamic=True):
    input_a = torch.randn((5, 32, 128), device=device, dtype=data_type)
    input_b = torch.randn((5, 128, 256), device=device, dtype=data_type)
    bias = None
    if func in [fn1, fn4, fn8, fn11]:
        bias = torch.randn((5, 1, 256), device=device, dtype=data_type)
    if func in [fn2, fn5, fn9, fn12]:
        residual = torch.randn((5, 32, 256), device=device, dtype=data_type)
        bias = residual

    res = func(input_a, input_b, bias)

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
    res1 = compiled(input_a, input_b, bias)
    is_similar(res.cpu().float(), res1.cpu().float())


class TestBMM:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [
            (fn0, True),
            (fn1, False),
            (fn2, True),
            (fn3, True),
            (fn4, False),
            (fn5, True),
            (fn6, False),
            (fn7, True),
            (fn8, True),
            (fn9, False),
            (fn10, True),
            (fn11, False),
            (fn12, True),
        ],
    )
    def test_bmm_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            bmm_test(self.xpu_graph_backend, pattern_func, dynamic)
        if pattern_func in [fn1, fn2, fn4, fn5, fn8, fn9, fn11, fn12]:
            assert "Pattern.FusedBAddBMM changed graph" in caplog.text
        else:
            assert "Pattern.FusedBAddBMM changed graph" not in caplog.text
        assert "Pattern.CustomBatchDenseLayer changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2, debug=True)
    bmm_test(xpu_graph_backend, fn0)
    bmm_test(xpu_graph_backend, fn1)
    bmm_test(xpu_graph_backend, fn2)
    bmm_test(xpu_graph_backend, fn3)
    bmm_test(xpu_graph_backend, fn4)
    bmm_test(xpu_graph_backend, fn5)
    bmm_test(xpu_graph_backend, fn6)
    bmm_test(xpu_graph_backend, fn7)
    bmm_test(xpu_graph_backend, fn8)
    bmm_test(xpu_graph_backend, fn9)
    bmm_test(xpu_graph_backend, fn10)
    bmm_test(xpu_graph_backend, fn11)
    bmm_test(xpu_graph_backend, fn12)
