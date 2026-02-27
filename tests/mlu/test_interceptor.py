import pytest
import torch
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.interceptor import OpInterceptor
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.test_utils import need_xpu_graph_logs

from tests.common.test_models import InplaceModel, compare_inference, compare_training

aten = torch.ops.aten

DISPATCH_ENV_CONFIGS = [
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
]

device = "mlu"
data_type = torch.float32


def test_op_monitor(caplog):
    def slow_add_tensor(a: torch.Tensor, b: torch.Tensor):
        a_np = a.numpy(force=True)
        b_np = b.numpy(force=True)
        c_np = a_np + b_np
        return torch.from_numpy(c_np)

    with need_xpu_graph_logs():
        with OpInterceptor({aten.add.Tensor: slow_add_tensor}, check_device=False, check_dtype=False):
            x = torch.randn((2, 2), device=device, dtype=data_type)
            y = torch.randn((2, 2), device=device, dtype=data_type)
            c = x + y
            print(c)

    assert f"Monitored op: {aten.add.Tensor}" in caplog.text and "diverges" not in caplog.text


def test_op_monitor_fail(caplog):
    with need_xpu_graph_logs():
        with OpInterceptor({aten.add.Tensor: aten.sub.Tensor}, check_device=False, check_dtype=False):
            x = torch.randn((2, 2), device=device, dtype=data_type)
            y = torch.randn((2, 2), device=device, dtype=data_type)
            c = x + y
            print(c)

    assert f"Monitored op: {aten.add.Tensor}" in caplog.text and "diverges" in caplog.text


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
class FaultyPattern(Pattern):
    def process(self, gm: torch.fx.GraphModule) -> bool:
        changed = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.sub.Tensor
                changed = True
        return changed


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestInferenceInterceptorUseGolden:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        request.cls.infer_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=1",
        )
        request.cls.faulty_pattern = FaultyPattern()
        request.cls.infer_backend.get_pattern_manager().register_pattern(request.cls.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_xfail_patterns(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_inference(device, data_type, ReproCls, self.infer_backend)
        assert "The inference pass diverges" in caplog.text


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestInferenceInterceptorUseActual:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        request.cls.infer_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=0",
        )
        request.cls.faulty_pattern = FaultyPattern()
        request.cls.infer_backend.get_pattern_manager().register_pattern(request.cls.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_xfail_patterns(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            with pytest.raises(AssertionError):
                compare_inference(device, data_type, ReproCls, self.infer_backend)
        assert "The inference pass diverges" in caplog.text


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestTrainingInterceptorUseGolden:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        request.cls.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=1",
        )
        request.cls.faulty_pattern = FaultyPattern()
        request.cls.train_backend.get_pattern_manager().register_pattern(request.cls.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_xfail_patterns(self, caplog, ReproCls, stage):
        with need_xpu_graph_logs():
            self.faulty_pattern._support_stages = [stage]
            compare_training(device, data_type, ReproCls, self.train_backend)
        if stage == FxStage.backward:
            assert "The backward pass diverges" in caplog.text
        else:
            assert "The forward pass diverges" in caplog.text


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestTrainingInterceptorUseActual:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        request.cls.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=0",
        )
        request.cls.faulty_pattern = FaultyPattern()
        request.cls.train_backend.get_pattern_manager().register_pattern(request.cls.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_xfail_patterns(self, caplog, ReproCls, stage):
        with need_xpu_graph_logs():
            self.faulty_pattern._support_stages = [stage]
            with pytest.raises(AssertionError):
                compare_training(device, data_type, ReproCls, self.train_backend)
        if stage == FxStage.backward:
            assert "The backward pass diverges" in caplog.text
        else:
            assert "The forward pass diverges" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, opt_level=OptLevel.level1, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )

    faulty_pattern = FaultyPattern()
    xpu_graph_backend.get_pattern_manager().register_pattern(faulty_pattern)

    faulty_pattern._support_stages = [FxStage.forward]
    compare_training(InplaceModel, xpu_graph_backend, nsteps=2)

    faulty_pattern._support_stages = [FxStage.backward]
    compare_training(InplaceModel, xpu_graph_backend, nsteps=2)
