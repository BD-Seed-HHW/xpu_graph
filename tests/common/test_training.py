import pytest
import torch
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs

from tests.common.test_models import SimpleModel, all_models, compare_training

DISPATCH_ENV_CONFIGS = [
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
]

device = "cpu"
data_type = torch.float32


@pytest.mark.env_settings(DISPATCH_ENV_CONFIGS)
class TestTraining:
    @pytest.fixture(autouse=True, scope="class")
    def setup_xpugraph(self, env_dispatch, request):
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level1, freeze=False)
        request.cls.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_training(self, ReproCls):
        compare_training(device, data_type, ReproCls, self.train_backend)


class TestTrainingWithPartiioner:
    @pytest.mark.parametrize(
        "partition_fn",
        [
            "MINCUT",
            "DEFAULT",
            "torch._functorch.partitioners.min_cut_rematerialization_partition",
            "dummy",
        ],
    )
    def test_training(self, caplog, monkeypatch, partition_fn):
        monkeypatch.setenv("XPUGRAPH_PARTITIONER", partition_fn)

        with need_xpu_graph_logs():
            train_config = xpu_graph.XpuGraphConfig(
                is_training=True,
                opt_level=OptLevel.level1,
                freeze=False,
                enable_interceptor="rtol=1e-6,atol=1e-5",
            )
            train_backend = xpu_graph.XpuGraph(train_config)
            compare_training(device, data_type, SimpleModel, train_backend)
        assert "Monitored forward" in caplog.text
        assert "Monitored backward" in caplog.text
        assert "diverges" not in caplog.text


if __name__ == "__main__":
    import os

    os.environ["XPUGRAPH_PARTITIONER"] = "torch._functorch.partitioners.min_cut_rematerialization_partition"
    train_config = xpu_graph.XpuGraphConfig(
        is_training=True, opt_level=OptLevel.level1, freeze=False, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    train_backend = xpu_graph.XpuGraph(train_config)
    compare_training(device, data_type, SimpleModel, train_backend)
