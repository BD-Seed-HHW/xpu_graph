import itertools
import os
import pickle

import pytest
import torch
from xpu_graph import OptLevel, mlu_compiler
from xpu_graph.test_utils import need_xpu_graph_logs

from tests.common.test_models import SimpleModel, compare_inference

device = "mlu"
dtype = torch.float32


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
def test_xpugraph_cache(caplog, tmp_path):
    # Note: currently, we do not guard on **freezed params**
    #       thus serialize a freeze artifact but with different parameters would result in wrong result
    infer_backend = mlu_compiler(is_training=False, opt_level=OptLevel.level2, freeze=False, cache=tmp_path)

    with need_xpu_graph_logs():
        for bsz in [8, 10, 20]:
            compare_inference(device, dtype, SimpleModel, infer_backend, bsz=bsz, dynamic=None)

        torch._dynamo.reset()

        for bsz in [8, 10, 20]:
            compare_inference(device, dtype, SimpleModel, infer_backend, bsz=bsz, dynamic=None)

    assert caplog.text.count("Save cache in location") == 2, "Should save cache twice"
    assert caplog.text.count("Use cache in location") == 2, "Should use cache twice"


@pytest.mark.parametrize("use_inductor", [True, False])
def test_xpugraph_inference_artifact_picklizable(caplog, tmp_path, use_inductor):
    if use_inductor:
        infer_backend = mlu_compiler(is_training=False, opt_level=OptLevel.level2, freeze=False, cache=tmp_path)
    else:
        infer_backend = mlu_compiler(
            is_training=False, opt_level=OptLevel.level2, freeze=False, cache=tmp_path, vendor_compiler_config=None
        )

    if infer_backend._config.fallback_legacy_dispatch is False:
        pytest.skip("Test only for Non AotAutograd mode")

    compiled_id = itertools.count()

    def pkl_compiler(gm, example_inputs):
        gm_id = next(compiled_id)
        pkl_path = tmp_path / f"compiled_artifact_{gm_id}.pkl"

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                loaded_artifact = pickle.load(f)
                return loaded_artifact.__call__

        compiled_artifact = infer_backend(gm, example_inputs)
        with open(pkl_path, "wb") as f:
            pickle.dump(compiled_artifact.__call__, f)
        return compiled_artifact

    with need_xpu_graph_logs():
        for bsz in [8, 10, 20]:
            compare_inference(device, dtype, SimpleModel, pkl_compiler, bsz=bsz, dynamic=None)

        # Mock a restart compilation process, but can access the pickled artifact
        torch._dynamo.reset()
        compiled_id = itertools.count()

        for bsz in [8, 10, 20]:
            compare_inference(device, dtype, SimpleModel, pkl_compiler, bsz=bsz, dynamic=None)

    assert caplog.text.count("Save cache in location") == 2, "Should save cache twice"
    assert caplog.text.count("Use cache in location") == 0, "Should directly use pickled artifact"


if __name__ == "__main__":
    test_xpugraph_cache(None, "/tmp/test_xpu_graph_cache")
