import pytest
import torch
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

from tests.common.test_models import all_models, compare_inference

DISPATCH_ENV_CONFIGS = [
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
]

device = "cpu"
data_type = torch.float32


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestInference:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, freeze=False)
        request.cls.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_inference(self, ReproCls):
        with skip_xpu_graph_cache(self.infer_backend):
            compare_inference(device, data_type, ReproCls, self.infer_backend)


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestFreezeInference:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        freeze_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, freeze=True)
        # Warning: DO NOT create both freeze and non-freeze in the same test case,
        request.cls.freeze_backend = xpu_graph.XpuGraph(freeze_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_freeze_inference(self, ReproCls):
        with skip_xpu_graph_cache(self.freeze_backend):
            compare_inference(device, data_type, ReproCls, self.freeze_backend)


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestInferenceWithInterceptor:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        infer_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2, freeze=False, enable_interceptor="rtol=1e-6,atol=1e-5"
        )
        request.cls.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_inference(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            compare_inference(device, data_type, ReproCls, self.infer_backend)
            assert "Monitored inference" in caplog.text
            assert "diverges" not in caplog.text


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestFreezeInferenceWithInterceptor:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        freeze_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2, freeze=True, enable_interceptor="rtol=1e-6,atol=1e-5"
        )
        # Warning: DO NOT create both freeze and non-freeze in the same test case,
        request.cls.freeze_backend = xpu_graph.XpuGraph(freeze_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_freeze_inference(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.freeze_backend):
            compare_inference(device, data_type, ReproCls, self.freeze_backend)
            assert "Monitored inference" in caplog.text
            assert "diverges" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, freeze=True, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_inference(device, data_type, ModCls, xpu_graph_backend)

    config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_inference(device, data_type, ModCls, xpu_graph_backend)
