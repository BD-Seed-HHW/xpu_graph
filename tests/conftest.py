import os

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo():
    try:
        torch._dynamo.reset()
        yield
    finally:
        torch._dynamo.reset()


@pytest.fixture(params=["0", "1"], scope="module", autouse=True)
def env_dispatch(request):
    try:
        orig_val = os.environ.get("XPUGRAPH_FALLBACK_LEGACY_DISPATCH", None)
        os.environ["XPUGRAPH_FALLBACK_LEGACY_DISPATCH"] = request.param
        yield request.param
    finally:
        if orig_val is not None:
            os.environ["XPUGRAPH_FALLBACK_LEGACY_DISPATCH"] = orig_val
        else:
            del os.environ["XPUGRAPH_FALLBACK_LEGACY_DISPATCH"]
