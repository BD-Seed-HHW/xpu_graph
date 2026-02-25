import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar

N0, N1, N2, N3, N4 = 10, 15, 255, 4, 128


def torch_func(unsqueeze_2, clamp_min, unsqueeze_3, clamp_min_1):
    expand = torch.ops.aten.expand.default(clamp_min, [N0, N1, 1, N3, N4])
    expand_1 = torch.ops.aten.expand.default(clamp_min_1, [N0, 1, N2, N3, N4])
    div = torch.ops.aten.div.Tensor(unsqueeze_2, expand)
    div_1 = torch.ops.aten.div.Tensor(unsqueeze_3, expand_1)
    mul_1 = torch.ops.aten.mul.Tensor(div, div_1)
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_1, [4])
    mul_2 = torch.ops.aten.mul.Tensor(sum_3, 0.5)
    return mul_2


def permute_sum_test(xpu_graph, func):
    DEV = "npu"
    DTYPE = torch.float32

    arg48_1 = torch.randn((N0, N1, N3, N4), dtype=DTYPE, device=DEV)
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg48_1, 2)
    arg49_1 = torch.randn((N0, N2, N3, N4), dtype=DTYPE, device=DEV)
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg49_1, 1)
    clamp_min = torch.randn((N0, N1, 1, N3, 1), dtype=DTYPE, device=DEV)
    clamp_min_1 = torch.randn((N0, 1, N2, N3, 1), dtype=DTYPE, device=DEV)

    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res_tri = compiled(unsqueeze_2, clamp_min, unsqueeze_3, clamp_min_1)
    res_ref = func(unsqueeze_2, clamp_min, unsqueeze_3, clamp_min_1)
    assert is_similar(res_ref.cpu(), res_tri.cpu())


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.npu_compiler(opt_level=OptLevel.level2)
    permute_sum_test(xpu_graph_backend, torch_func)
