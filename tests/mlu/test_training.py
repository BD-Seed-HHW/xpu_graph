
import pytest
import torch
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs

from tests.common.test_models import all_models, compare_training

device = "mlu"
data_type = torch.float32


class TestTraining:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level1,
            freeze=False,
            vendor_compiler_config=None,  # FIXME: inductor has some bug with index_put
            cache=xpu_graph.cache.no_cache(),
        )

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_training(self, ReproCls):
        compare_training(device, data_type, ReproCls, self.train_backend)


class TestTrainingWithInterceptor:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level1,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5",
            vendor_compiler_config=None,  # FIXME: inductor has some bug with index_put
            cache=xpu_graph.cache.no_cache(),
        )

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_training(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_training(device, data_type, ReproCls, self.train_backend)
        assert "Monitored forward" in caplog.text
        assert "Monitored backward" in caplog.text
        assert "diverges" not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, opt_level=OptLevel.level1, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    for ModCls in all_models:
        compare_training(device, data_type, ModCls, xpu_graph_backend)
