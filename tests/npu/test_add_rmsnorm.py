import torch
import torch._dynamo.config
from torch import nn
from xpu_graph import OptLevel, Target, XpuGraph, XpuGraphConfig
from xpu_graph.accuracy_utils import assert_close, benchmark_compare_close
from xpu_graph.test_utils import need_xpu_graph_logs


def call_aten_kernel(input, residual, weight):
    new_residual_tensor = input.to(torch.float32) + residual.to(torch.float32)
    new_residual = new_residual_tensor.to(torch.bfloat16)
    eps = 1e-6
    x = new_residual
    rms = torch.sqrt((x.pow(2)).mean(-1, keepdim=True) + eps)  # 计算均方根
    x_norm = x / rms  # 标准化
    y_ref = weight * x_norm
    y_ref = y_ref.to(input.dtype)
    return y_ref


class NPU_RMSNormWithResidual(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, input, residual, arg486_1, arg487_1, weight=None):
        """
        1. 将输入和残差转换为float32并相加
        2. 转回bfloat16精度
        3. 应用NPU RMSNorm
        """
        quant_matmul_v2_111 = input
        npu_dtype_cast_164 = residual
        arg485_1 = weight
        npu_dtype_cast_165 = torch.ops.npu.npu_dtype_cast.default(quant_matmul_v2_111, torch.float32)
        quant_matmul_v2_111 = None
        npu_dtype_cast_166 = torch.ops.npu.npu_dtype_cast.default(npu_dtype_cast_164, torch.float32)
        npu_dtype_cast_164 = None
        add_55 = torch.ops.aten.add.Tensor(npu_dtype_cast_165, npu_dtype_cast_166)
        npu_dtype_cast_165 = npu_dtype_cast_166 = None
        npu_dtype_cast_167 = torch.ops.npu.npu_dtype_cast.default(add_55, torch.bfloat16)
        add_55 = None
        npu_rms_norm_56 = torch.ops.npu.npu_rms_norm.default(npu_dtype_cast_167, arg485_1)
        npu_dtype_cast_167 = arg485_1 = None
        getitem_476 = npu_rms_norm_56[0]
        return getitem_476


def compare_add_rmsnorm_pattern(xpu_graph_backend):
    # create input data
    shape = [1, 3584]
    dtype = torch.bfloat16
    input = torch.randn(shape, dtype=dtype).npu()
    input = input.clamp(0, 10)
    residual = torch.randn(shape, dtype=dtype).npu()
    weight = torch.randn(shape, dtype=dtype).npu()
    arg486_1 = torch.tensor([0], dtype=torch.int32).npu()
    arg487_1 = torch.randn(size=(152064, 3584), dtype=torch.bfloat16).npu()

    # init our graph
    model = NPU_RMSNormWithResidual(weight).npu()
    model_forward = model.forward
    compiled_model = torch.compile(model_forward, backend=xpu_graph_backend, dynamic=False)

    # get result
    mm_out = compiled_model(input, residual, arg486_1, arg487_1, weight)

    mm_ref_fp32 = model(
        input.to(torch.float32),
        residual.to(torch.float32),
        arg486_1,
        arg487_1.to(torch.float32),
        weight.to(torch.float32),
    )
    mm_ref_bf16 = call_aten_kernel(input, residual, weight)

    try:
        assert_close(mm_ref_fp32.to(torch.float32), mm_out)
    except Exception as e:
        print(e)
        print("starting benchmark compare_close:")
        benchmark_compare_close(mm_ref_fp32.to(torch.float32), mm_out, mm_ref_bf16)
        print("PASSED")


class TestAddRMSNormPattern:
    def setup_class(self):
        config = XpuGraphConfig(
            is_training=False,
            freeze=True,
            target=Target.npu,
            opt_level=OptLevel.level2,
            vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
            debug=False,
        )
        self.xpu_graph_backend = XpuGraph(config)

    def test_add_rmsnorm_pattern(self, caplog):
        with need_xpu_graph_logs():
            compare_add_rmsnorm_pattern(self.xpu_graph_backend)
        assert "Pattern.FusedAddRmsnorm changed graph" in caplog.text
        assert "Save cache in location" not in caplog.text


if __name__ == "__main__":
    config = XpuGraphConfig(
        is_training=False,
        dump_graph=True,
        freeze=True,
        target=Target.npu,
        opt_level=OptLevel.level2,
        vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
        debug=True,
    )
    xpu_graph_backend = XpuGraph(config)
    compare_add_rmsnorm_pattern(xpu_graph_backend)
