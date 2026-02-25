import pytest
import torch
import torch.nn.functional as F
import xpu_graph
from xpu_graph.test_utils import is_similar

aten = torch.ops.aten
device = "mlu:0"
dtype = torch.float16

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(ffn_input, ffn_weight1, ffn_weight2, ffn_weight3, bias1, bias2, bias3, act1, act2, act3):
    output = torch.matmul(ffn_input, ffn_weight1)
    if bias1 is not None:
        output = output + bias1
    if act1 != "none":
        output = F.relu(output)
    output = torch.matmul(output, ffn_weight2)
    if bias2 is not None:
        output = output + bias2
    if act2 != "none":
        output = F.relu(output)
    output = torch.matmul(output, ffn_weight3)
    if bias3 is not None:
        output = output + bias3
    if act3 != "none":
        output = F.relu(output)

    return output


def serial_mm_test(xpu_graph_backend, func, dynamic):
    with torch.no_grad():
        batch = 16
        input_size = 20
        weight_size1 = 32
        weight_size2 = 16
        weight_size3 = 1

        input = torch.randn(batch, input_size, dtype=dtype, device=device)
        weight1 = torch.randn(input_size, weight_size1, dtype=dtype, device=device)
        weight2 = torch.randn(weight_size1, weight_size2, dtype=dtype, device=device)
        weight3 = torch.randn(weight_size2, weight_size3, dtype=dtype, device=device)

        bias1 = torch.randn(weight_size1, dtype=dtype, device=device)
        bias2 = torch.randn(weight_size2, dtype=dtype, device=device)
        bias3 = torch.randn(weight_size3, dtype=dtype, device=device)
        act1 = "relu"
        act2 = "relu"
        act3 = "none"

        args = [input]
        args += [
            weight1,
            weight2,
            weight3,
            bias1,
            bias2,
            bias3,
            act1,
            act2,
            act3,
        ]

        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
        res1 = func(*args)
        res = compiled(*args)

        is_similar(res1.cpu().float(), res.cpu().float())


class TestSerialMM:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    @pytest.mark.parametrize("dynamic", [True, False])
    def test_serial_mm_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            serial_mm_test(self.xpu_graph_backend, pattern_func, dynamic)
        assert "Pattern.FusedDenseTower2 changed graph" in caplog.text
        assert "Pattern.FusedDenseTower3 changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2)
    serial_mm_test(xpu_graph_backend, fn0)
