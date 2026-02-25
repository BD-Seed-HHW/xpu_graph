import math

import pytest
import torch
import triton
import triton.language as tl
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache
from xpu_graph.utils import __XPU_GRAPH_ENVS__


@triton.jit
def simple_scale(x_ptr, y_ptr, n_rows, scale, ROW_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    for i in range(n_rows):
        block_start = pid * BLOCK_SIZE + i * ROW_SIZE
        x = tl.load(x_ptr + block_start + tl.arange(0, BLOCK_SIZE))
        x = x * scale
        tl.store(y_ptr + block_start + tl.arange(0, BLOCK_SIZE), x)


def fn0(x, scale):
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[1], meta["BLOCK_SIZE"]),)
    simple_scale[grid](x, y, n_rows=x.shape[0], scale=scale, ROW_SIZE=x.shape[1], BLOCK_SIZE=128)
    return y


def ref_fn0(x, scale):
    return x * scale


def compare_func(func, ref_func, backend):
    compiled = torch.compile(func, backend=backend)

    for bsz in [2, 8, 10]:
        x = torch.randn(bsz, 1024, device="mlu")
        scale = 1 / math.sqrt(bsz)
        y = compiled(x, scale)
        ref = func(x, scale)
        ref2 = ref_func(x, scale)
        assert torch.allclose(ref, ref2)
        assert torch.allclose(y, ref2)


class TestTritonWrap:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False)

    @pytest.mark.parametrize(
        "orig_func, ref_func",
        [
            (fn0, ref_fn0),
        ],
    )
    def test_triton_wrap(self, monkeypatch, caplog, orig_func, ref_func):
        import torch._dynamo.config

        monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", True)
        monkeypatch.setenv(__XPU_GRAPH_ENVS__.dispatch_tensorify_python_scalars, "True")

        with skip_xpu_graph_cache(self.infer_backend), need_xpu_graph_logs():
            compare_func(orig_func, ref_func, self.infer_backend)
        assert caplog.text.count("mlu_compile start...") == 3


if __name__ == "__main__":
    import torch._dynamo.config

    torch._dynamo.config.capture_scalar_outputs = True
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, debug=True)
    compare_func(fn0, ref_fn0, xpu_graph_backend)
