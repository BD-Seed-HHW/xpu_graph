import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar


def fn0(view_7, gather, gather_1, arg107_1, full_default):
    slice_3: "i64[11, 256]" = torch.ops.aten.slice.Tensor(arg107_1, 1, 0, 512)
    unsqueeze_1: "i64[11, 1, 256]" = torch.ops.aten.unsqueeze.default(slice_3, 1)
    unsqueeze_2: "i64[11, 1, 1, 256]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2)
    expand: "i64[11, 1, 256, 256]" = torch.ops.aten.expand.default(unsqueeze_2, [11, 1, 256, 256])
    npu_dtype_cast_1: "f16[11, 1, 256, 256]" = torch.ops.npu.npu_dtype_cast.default(expand, torch.float16)
    sub_1: "f16[11, 1, 256, 256]" = torch.ops.aten.sub.Tensor(1.0, npu_dtype_cast_1)
    npu_dtype_cast_2: "b8[11, 1, 256, 256]" = torch.ops.npu.npu_dtype_cast.default(sub_1, torch.bool)
    where: "f16[11, 1, 256, 256]" = torch.ops.aten.where.self(npu_dtype_cast_2, full_default, sub_1)
    permute_10: "f16[11, 12, 256, 256]" = torch.ops.aten.permute.default(gather_1, [0, 1, 3, 2])
    add_4: "f16[11, 12, 256, 256]" = torch.ops.aten.add.Tensor(gather, permute_10)
    mul_4: "f16[11, 12, 256, 256]" = torch.ops.aten.mul.Tensor(add_4, 0.07216878364870322)
    add_5: "f16[11, 12, 256, 256]" = torch.ops.aten.add.Tensor(view_7, mul_4)
    add_6: "f16[11, 12, 256, 256]" = torch.ops.aten.add.Tensor(add_5, where)
    convert_element_type_20: "f32[11, 12, 256, 256]" = torch.ops.prims.convert_element_type.default(
        add_6, torch.float32
    )
    amax: "f32[11, 12, 256, 1]" = torch.ops.aten.amax.default(convert_element_type_20, [-1], True)
    sub_4: "f32[11, 12, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax)
    exp: "f32[11, 12, 256, 256]" = torch.ops.aten.exp.default(sub_4)
    sum_1: "f32[11, 12, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[11, 12, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1)
    convert_element_type_21: "f16[11, 12, 256, 256]" = torch.ops.prims.convert_element_type.default(div, torch.float16)
    return convert_element_type_21


def permute_sum_test(xpu_graph, func):
    data_type = torch.float16
    buf34 = torch.rand([132, 256, 256], dtype=data_type).npu()
    buf47 = torch.rand([11, 12, 256, 256], dtype=data_type).npu()
    buf59 = torch.rand([11, 12, 256, 256], dtype=data_type).npu()
    arg107_1 = torch.randint(1, 1000, (11, 256), dtype=torch.int64).npu()
    buf61 = torch.full([], -65504.0, dtype=data_type).npu()
    buf73 = torch.zeros([11, 12, 256, 256], dtype=data_type).npu()
    view_7 = torch.ops.aten.view.default(buf34, [11, 12, 256, 256])

    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res_tri = compiled(view_7, buf47, buf59, arg107_1, buf61)
    res_ref = func(view_7, buf47, buf59, arg107_1, buf61)
    assert is_similar(res_ref.cpu(), res_tri.cpu())


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.npu_compiler(opt_level=OptLevel.level2)
    permute_sum_test(xpu_graph_backend, fn0)
