import numpy as np
import torch
import xpu_graph

device = "mlu:0"


class NumpyScaleModule(torch.nn.Module):
    def __init__(self, input_dim):
        super(NumpyScaleModule, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return x / np.sqrt(float(self.input_dim))


def compare_test(xpugraph_backend):
    input_dim = 1024
    input_tensor = torch.randn(1, input_dim, device=device)
    orig = NumpyScaleModule(input_dim)
    compiled = NumpyScaleModule(input_dim)

    compiled.forward = torch.compile(compiled.forward, backend=xpugraph_backend, dynamic=False)

    orig_output = orig(input_tensor)
    compiled_output = compiled(input_tensor)
    assert torch.allclose(orig_output, compiled_output, atol=1e-6)


class TestNumpyFloat:
    def setup_class(self):
        self.xpugraph_backend = xpu_graph.mlu_compiler(is_training=False, constant_folding=True, debug=True)

    def test_numpy_float(self):
        compare_test(self.xpugraph_backend)


if __name__ == "__main__":
    compare_test("inductor")
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, constant_folding=True, debug=True)
    compare_test(xpu_graph_backend)
