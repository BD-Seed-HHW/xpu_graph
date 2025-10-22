import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def fn_linear(x, w, b):
    return torch.nn.functional.linear(x, w.T, b)


def compare_linear_to_mm(test_func, xpu_graph_backend, bias=True):
    compiled = torch.compile(test_func, backend=xpu_graph_backend, dynamic=None)

    weight_tensor = torch.rand(256, 128).to("mlu")
    if bias:
        bias_tensor = torch.rand(128).to("mlu")
    else:
        bias_tensor = None

    for bsz in [8, 32, 80]:
        input_tensor = torch.rand(bsz, 256).to("mlu")
        result = compiled(input_tensor, weight_tensor, bias_tensor)
        ref = fn_linear(input_tensor, weight_tensor, bias_tensor)
        ref2 = (
            torch.addmm(bias_tensor, input_tensor, weight_tensor) if bias else torch.matmul(input_tensor, weight_tensor)
        )
        assert is_similar(result, ref)
        assert is_similar(result, ref2)


class TestPluginPattern:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            debug=False,
            vendor_compiler_config=None,
        )

    @pytest.mark.parametrize("bias", [True, False])
    def test_linear_bias(self, caplog, bias):
        with need_xpu_graph_logs():
            compare_linear_to_mm(fn_linear, self.xpu_graph_backend, bias)
        if bias:
            assert (
                caplog.text.count(
                    "Pattern.xpu_graph.passes.patterns.targets.mlu.linear_to_matmul-linear_pattern2 changed graph"
                )
                == 2
            )
        else:
            assert (
                caplog.text.count(
                    "Pattern.xpu_graph.passes.patterns.targets.mlu.linear_to_matmul-linear_pattern1 changed graph"
                )
                == 2
            )


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
        debug=True,
        vendor_compiler_config=None,
    )
    compare_linear_to_mm(fn_linear, xpu_graph_backend)
