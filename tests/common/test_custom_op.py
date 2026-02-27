import pytest
import torch
import torch.nn as nn
import xpu_graph
from torch.utils.checkpoint import checkpoint
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs

device = "cpu"
dtype = torch.float32


@torch.library.custom_op("test_op::numpy_mul", mutates_args=())
def numpy_mul(x: torch.Tensor, y: float) -> torch.Tensor:
    x_np = x.numpy(force=True)
    z_np = x_np * y
    return torch.from_numpy(z_np).to(x.device)


@numpy_mul.register_fake
def _(x, y):
    return torch.empty_like(x)


def setup_context(ctx, inputs, output):
    (
        x,
        y,
    ) = inputs
    ctx.y = y


def backward(ctx, grad):
    return grad * ctx.y, None


numpy_mul.register_autograd(backward, setup_context=setup_context)


class SimpleWithCustomOp(torch.nn.Module):
    def __init__(self, input_dim, p=0.1):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, input_dim)
        self.scale = p

    def forward(self, x):
        y = self.fc(x)
        z = numpy_mul(y, self.scale)
        return z.sum(dim=-1)


class NumpyMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float) -> torch.Tensor:
        x_np = x.numpy(force=True)
        z_np = x_np * y
        ctx.save_for_backward(torch.tensor(y))
        return torch.from_numpy(z_np).to(x.device)

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return grad * y, None


numpy_mul2 = NumpyMul.apply


class SimpleWithAutogradFunction(torch.nn.Module):
    def __init__(self, input_dim, p=0.1):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, input_dim)
        self.scale = p

    def forward(self, x):
        y = self.fc(x)
        z = numpy_mul2(y, self.scale)
        z = numpy_mul2(z, self.scale)
        return z.sum(dim=-1)


class SimpleSwiGLUwithCKPT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, 2 * input_dim, bias=True)

    def _compute_swiglu(self, x):
        projected = self.projection(x)
        a, b = projected.split(self.input_dim, dim=-1)
        return nn.functional.silu(a) * b

    def forward(self, x):
        if self.training:
            y = checkpoint(self._compute_swiglu, x, use_reentrant=False)
        else:
            y = self._compute_swiglu(x)

        return y.sum(dim=-1)


def compare_training_with_custom_op(ModCls, backend, nsteps=4, bsz=8, input_dim=16):
    golden = ModCls(input_dim)
    compiled = ModCls(input_dim)
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)

    compiled.load_state_dict(golden.state_dict())
    input = torch.randn((bsz, input_dim), device=device, dtype=dtype)
    target = torch.randn((bsz,), device=device, dtype=dtype)

    optimizer_golden = torch.optim.AdamW(golden.parameters())
    optimizer_compiled = torch.optim.AdamW(compiled.parameters())
    optimizer_compiled.load_state_dict(optimizer_golden.state_dict())
    print(optimizer_golden.state_dict(), optimizer_compiled.state_dict())

    loss_fn = nn.MSELoss()

    for i in range(nsteps):
        optimizer_golden.zero_grad()
        loss_golden = loss_fn(golden(input), target)
        loss_golden.backward()
        optimizer_golden.step()

        optimizer_compiled.zero_grad()
        loss_compiled = loss_fn(compiled(input), target)
        loss_compiled.backward()
        optimizer_compiled.step()

        print(f"Step: {i} golden: {loss_golden}, compiled: {loss_compiled}")
        assert is_similar(loss_golden, loss_compiled)
        for p_name, p_golden in golden.named_parameters():
            p_compiled = compiled.get_parameter(p_name)
            assert is_similar(p_golden, p_compiled)


class TestCustomFallbackTraining:
    def setup_class(self):
        train_config = xpu_graph.XpuGraphConfig(
            is_training=True, opt_level=OptLevel.level2, freeze=False, fallback_legacy_dispatch=True
        )
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "ReproCls",
        [
            SimpleWithCustomOp,
            SimpleWithAutogradFunction,
            SimpleSwiGLUwithCKPT,
        ],
    )
    def test_custom_op(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_training_with_custom_op(ReproCls, self.train_backend)
        if self.train_backend._config.fallback_legacy_dispatch:
            if ReproCls == SimpleWithCustomOp:
                assert "Higher order operators detected" not in caplog.text
            else:
                assert "Higher order operators detected" in caplog.text


def compare_inference_with_custom_op(ModCls, backend, nbatches=4, bsz=8, input_dim=16):
    golden = ModCls(input_dim).eval()
    compiled = ModCls(input_dim).eval()
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)

    compiled.load_state_dict(golden.state_dict())

    with torch.no_grad():
        for i in range(nbatches):
            input = torch.randn((bsz, input_dim), device=device, dtype=dtype)
            golden_output = golden(input)
            compiled_output = compiled(input)
            print(f"Batch: {i} golden: {golden_output}, compiled: {compiled_output}")
            assert is_similar(golden_output, compiled_output)


class TestCustomFallbackInference:
    def setup_class(self):
        inference_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2, freeze=False, fallback_legacy_dispatch=True
        )
        self.inference_backend = xpu_graph.XpuGraph(inference_config)

    @pytest.mark.parametrize(
        "ReproCls",
        [
            SimpleWithCustomOp,
            SimpleWithAutogradFunction,
            SimpleSwiGLUwithCKPT,
        ],
    )
    def test_custom_op(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_inference_with_custom_op(ReproCls, self.inference_backend)
        assert "Higher order operators detected" not in caplog.text
        if self.inference_backend._config.fallback_legacy_dispatch:
            assert "decompose graph complete" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=True, opt_level=OptLevel.level2, freeze=False, debug=True, fallback_legacy_dispatch=False
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in [SimpleSwiGLUwithCKPT]:
        compare_training_with_custom_op(ModCls, xpu_graph_backend)
