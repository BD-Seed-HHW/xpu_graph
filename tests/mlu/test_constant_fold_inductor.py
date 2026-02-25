import torch
from xpu_graph import mlu_compiler
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu"
dtype = torch.float32


def constant_folding_with_reload_test(xpu_graph_backend):
    class CanConstantFolding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand((128, 128), device=device, dtype=dtype))

        @torch.no_grad()
        def forward(self, x):
            weight = torch.relu(torch.relu(self.weight))
            bias = torch.tensor([1] * 128, device=device) + torch.tensor([1], device=device)
            return torch.matmul(x + bias, weight)

    mod = CanConstantFolding().mlu()
    compiled_mod = CanConstantFolding().mlu()
    compiled_mod.load_state_dict(mod.state_dict())

    with need_xpu_graph_logs(), skip_xpu_graph_cache(xpu_graph_backend):
        compiled_mod.forward = torch.compile(compiled_mod.forward, backend=xpu_graph_backend, dynamic=False)
        res = compiled_mod(torch.ones((128, 128), device=device, dtype=dtype))
        expect = mod(torch.ones((128, 128), device=device, dtype=dtype))
        assert is_similar(res, expect)

        state_dict = {"weight": mod.weight + 1}
        mod.load_state_dict(state_dict)
        expect = mod(torch.ones((128, 128), device=device, dtype=dtype))
        compiled_mod.load_state_dict(state_dict)
        res = compiled_mod(torch.ones((128, 128), device=device, dtype=dtype))
        assert is_similar(res, expect)

        # Mock custom param reload
        state_dict = {"weight": mod.weight + 1}
        with torch.no_grad():
            for name, param in mod.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
        expect = mod(torch.ones((128, 128), device=device, dtype=dtype))
        with torch.no_grad():
            for name, param in compiled_mod.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
        res = compiled_mod(torch.ones((128, 128), device=device, dtype=dtype))
        assert is_similar(res, expect)


class TestConstantFolding:
    def setup_class(self):
        self.xpu_graph_backend = mlu_compiler(
            is_training=False,
            freeze=True,
            constant_folding=True,
            folding_freezed_params=False,
            vendor_compiler_config={"mode": "default"},
        )

    def test_constant_folding(self, caplog):
        constant_folding_with_reload_test(self.xpu_graph_backend)
        assert "Optimizer.ConstantFolding changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = mlu_compiler(
        is_training=False,
        freeze=True,
        constant_folding=True,
        folding_freezed_params=False,
        vendor_compiler_config={"mode": "default"},
        debug=True,
    )
    constant_folding_with_reload_test(xpu_graph_backend)
