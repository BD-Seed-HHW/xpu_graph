import pytest
import torch
from xpu_graph import OptLevel, XpuGraph, XpuGraphConfig, XpuGraphLocalCache
from xpu_graph.test_utils import need_xpu_graph_logs

from tests.test_models import SimpleModel, compare_inference
from tests.utils import parametrize_class_env

device = "cpu"
dtype = torch.float32


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
@pytest.mark.parametrize("freezing", [False, True])
def test_xpugraph_cache(caplog, tmp_path, freezing):
    # Note: currently, we do not guard on **freezed params**
    #       thus serialize a freeze artifact but with different parameters would result in wrong result
    infer_config = XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, freeze=freezing)
    infer_backend = XpuGraph(infer_config, XpuGraphLocalCache(tmp_path))

    input_dim = 16
    ref_model = SimpleModel(input_dim).to(device=device, dtype=dtype).eval()
    with need_xpu_graph_logs():
        for bsz in [8, 10, 20]:
            compare_inference(
                device,
                dtype,
                SimpleModel,
                infer_backend,
                bsz=bsz,
                input_dim=input_dim,
                dynamic=None,
                state_dict=ref_model.state_dict(),
            )

        torch._dynamo.reset()

        for bsz in [8, 10, 20]:
            compare_inference(
                device,
                dtype,
                SimpleModel,
                infer_backend,
                bsz=bsz,
                input_dim=input_dim,
                dynamic=None,
                state_dict=ref_model.state_dict(),
            )

    assert caplog.text.count("Save cache in location") == 2, "Should save cache twice"
    if freezing:
        # Note(chenyifan):
        #   Freezing with different parameters (by object id) would always trigger RECOMPILE
        #   Currently, it will use the same cache, BE CAREFUL if parameter changes
        assert caplog.text.count("Use cache in location") == 4, "Should use cache twice"
    else:
        assert caplog.text.count("Use cache in location") == 2, "Should use cache twice"


if __name__ == "__main__":
    test_xpugraph_cache(None, "/tmp/test_xpu_graph_cache", freezing=True)
