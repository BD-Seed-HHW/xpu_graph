import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "npu"
data_type = torch.float16


class NpuQuantMatmulModule(torch.nn.Module):
    def __init__(self, weight_shape):
        super().__init__()

        x2_data = torch.randint(0, 29, weight_shape, dtype=torch.int8, device=device)
        self.register_buffer("x2_data", x2_data, persistent=True)

    def forward(self, x1, scale):
        return torch.ops.npu.npu_quant_matmul.default(x1, self.x2_data, scale)


def create_test_data(func_name, weight_shape, batch_size=32):
    if func_name == "npu_quant_matmul":
        K, N = weight_shape

        x1_shape = (batch_size, K)
        scale_shape = (N,)

        x1 = torch.randint(0, 8, x1_shape, dtype=torch.int8, device=device)
        scale = torch.randn(scale_shape, dtype=torch.float32, device=device)

        module = NpuQuantMatmulModule(weight_shape).to(device)
        return module, (x1, scale)

    else:
        raise ValueError(f"Unknown function name: {func_name}")


def run_nd_to_nz_test(xpu_graph_backend, func_name, weight_shape, expected_pattern=None):
    module, input_args = create_test_data(func_name, weight_shape)

    with torch.no_grad():
        module.eval()

        result_direct = module(*input_args)

        compiled_module = torch.compile(module, backend=xpu_graph_backend, dynamic=False)
        result_compiled = compiled_module(*input_args)

        if isinstance(result_direct, (list, tuple)):
            assert len(result_direct) == len(
                result_compiled
            ), f"Output length mismatch: {len(result_direct)} vs {len(result_compiled)}"

            for i, (direct, compiled) in enumerate(zip(result_direct, result_compiled)):
                assert is_similar(
                    direct.cpu(), compiled.cpu(), rtol=1e-2, atol=1e-2
                ), f"Output {i} mismatch: max diff = {torch.max(torch.abs(direct - compiled)).item()}"
        else:
            assert is_similar(
                result_direct.cpu(), result_compiled.cpu(), rtol=1e-2, atol=1e-2
            ), f"Result mismatch: max diff = {torch.max(torch.abs(result_direct - result_compiled)).item()}"


class TestFoldNdToNzFormat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.XpuGraph(
            xpu_graph.XpuGraphConfig(
                is_training=False,
                debug=True,
                target=xpu_graph.Target.npu,
                freeze=True,
                opt_level=OptLevel.level1,
                vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
            )
        )

    @pytest.mark.parametrize(
        "weight_shape",
        [
            pytest.param(torch.Size([18944, 3584]), id="large_k_small_n"),
            pytest.param(torch.Size([3584, 18944]), id="small_k_large_n"),
        ],
    )
    def test_npu_quant_matmul_nd_to_nz(self, caplog, weight_shape):
        print(f"\nTesting with weight_shape: {weight_shape}")
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "npu_quant_matmul", weight_shape)

        assert "Pattern.FoldNdToNzFormat changed graph" in caplog.text
