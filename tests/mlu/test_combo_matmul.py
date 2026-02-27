import pytest
import torch
import torch.nn.functional as F
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
data_type = torch.float32


def fn0(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight in weight_list:
        outputs_original.append(inputs @ weight)
    return outputs_original


def fn1(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight in weight_list:
        outputs_original.append(torch.bmm(inputs, weight))
    return outputs_original


def fn2(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(inputs @ weight + bias)
    return outputs_original


def fn3(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(inputs @ weight + bias)
    return outputs_original


def fn4(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(inputs @ weight + bias)
    return outputs_original


def fn5(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(torch.relu(inputs @ weight + bias))
    return outputs_original


def fn6(inputs, weight_list, bias_list):
    input_list = inputs.split(inputs.shape[0] // len(weight_list), dim=0)
    input_list = [i.squeeze(0) for i in input_list]
    outputs_original = []
    for input, weight in zip(input_list, weight_list):
        outputs_original.append(torch.relu(input @ weight))
    return outputs_original


def fn7(inputs, weight_list, bias_list):
    input_list = inputs.split(inputs.shape[0] // len(weight_list), dim=0)
    input_list = [i.squeeze(0) for i in input_list]
    outputs_original = []
    for input, weight in zip(input_list, weight_list):
        outputs_original.append(torch.relu(F.linear(input, weight.T)))
    return outputs_original


def fn8(inputs, weight_list, bias_list):
    weight_list = [w.t() for w in weight_list]
    return fn0(inputs, weight_list, bias_list)


def create_input(func, M, N, K):
    T = 8
    inputs = None
    weight_list = None
    bias_list = None
    if func in [fn0, fn2, fn4, fn5, fn8]:
        M, N, K = 5, 8, 7
        inputs = torch.randn((1, M, N), device=device, dtype=data_type)
        weight_list = [torch.randn((N, K), device=device, dtype=data_type) for _ in range(T)]
        if func == fn4:
            bias_list = [torch.randn((K), device=device, dtype=data_type) for _ in range(T)]
        else:
            bias_list = [torch.randn((M, K), device=device, dtype=data_type) for _ in range(T)]
        if func == fn8:
            weight_list = [w.t().contiguous() for w in weight_list]

    if func in [fn1, fn3]:
        S = 3
        M, N, K = 5, 6, 7
        inputs = torch.randn((S, M, N), device=device, dtype=data_type)
        weight_list = [torch.randn((S, N, K), device=device, dtype=data_type) for _ in range(T)]
        bias_list = [torch.randn((S, M, K), device=device, dtype=data_type) for _ in range(T)]
    if func in [fn6, fn7]:
        M, N, K = 5, 8, 7
        inputs = torch.randn((T, M, N), device=device, dtype=data_type)
        weight_list = [torch.randn((N, K), device=device, dtype=data_type) for _ in range(T)]
        bias_list = None
    return inputs, weight_list, bias_list


def combine_matmul_test(xpu_graph_backend, func, dynamic=True):
    inputs, weight_list, bias_list = create_input(func, 5, 8, 7)
    res = func(inputs, weight_list, bias_list)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
    res1 = compiled(inputs, weight_list, bias_list)
    for i in range(len(weight_list)):
        assertTensorsEqual(
            res[i].cpu().float(),
            res1[i].cpu().float(),
            0.005,
            use_MSE=True,
            use_RAE=True,
        )


def combine_matmul_test_with_loss_and_grad(xpu_graph_backend, func, dynamic=True):
    inputs, weight_list, bias_list = create_input(func, 5, 8, 7)

    inputs = inputs.requires_grad_(True)
    weight_list = [w.requires_grad_(True) for w in weight_list]
    if bias_list is not None:
        bias_list = [b.requires_grad_(True) for b in bias_list]

    with torch.no_grad():
        temp_outputs = func(inputs, weight_list, bias_list)
    grad_outputs = [torch.randn_like(output, device=device, dtype=data_type) for output in temp_outputs]

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)

    grad_params = [inputs] + weight_list
    if bias_list is not None:
        grad_params += bias_list

    res0 = compiled(inputs, weight_list, bias_list)
    grads0 = torch.autograd.grad(res0, grad_params, grad_outputs, allow_unused=True)

    res1 = func(inputs, weight_list, bias_list)
    grads1 = torch.autograd.grad(res1, grad_params, grad_outputs, allow_unused=True)

    assert len(res0) == len(res1), f"输出长度不匹配: {len(res0)} vs {len(res1)}"
    for i in range(len(res0)):
        assertTensorsEqual(
            res0[i].cpu().float(),
            res1[i].cpu().float(),
            0.005,
            use_MSE=True,
            use_RAE=True,
        )

    assert len(grads0) == len(grads1), f"梯度数量不匹配: {len(grads0)} vs {len(grads1)}"
    for i, (grad0, grad1) in enumerate(zip(grads0, grads1)):
        if grad0 is not None and grad1 is not None:
            assertTensorsEqual(
                grad0.cpu().float(),
                grad1.cpu().float(),
                0.005,
                use_MSE=True,
                use_RAE=True,
            )
        elif grad0 is None and grad1 is None:
            continue  # 都是None，正常
        else:
            raise AssertionError(f"梯度 {i} 的存在性不匹配: {grad0 is not None} vs {grad1 is not None}")


class TestCombineMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level2,
        )

    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
            fn5,
            fn6,
            fn7,
            fn8,
        ],
    )
    def test_matmul_patterns_inference(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_matmul_test(self.xpu_graph_backend, pattern_func, dynamic)
            assert "Pattern.FusedCombineMatMul changed graph" in caplog.text

    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
            fn5,
            fn6,
            fn7,
            fn8,
        ],
    )
    @pytest.mark.skip(reason="combo_mm for training is not fully verified yet, disable it for now")
    def test_matmul_patterns_training(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            combine_matmul_test_with_loss_and_grad(self.train_backend, pattern_func, dynamic)
            assert "Pattern.FusedCombineMatMul changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True,
        opt_level=OptLevel.level2,
        debug=True,
        vendor_compiler_config=None,
    )
    combine_matmul_test_with_loss_and_grad(xpu_graph_backend, fn6, dynamic=True)

    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
        opt_level=OptLevel.level2,
        debug=True,
        vendor_compiler_config=None,
    )
    combine_matmul_test(xpu_graph_backend, fn8, dynamic=False)
