
import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo():
    try:
        torch._dynamo.reset()
        yield
    finally:
        torch._dynamo.reset()


DEFAULT_ENV_CONFIGS = [{}]


@pytest.fixture(scope="class")
def env_dispatch(request):
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    for k, v in request.param.items():
        m.setenv(k, v)

    yield
    m.undo()


def pytest_generate_tests(metafunc):
    if "env_dispatch" in metafunc.fixturenames:
        marker = metafunc.definition.get_closest_marker("env_settings")
        env_configs = marker.args[0] if marker is not None else DEFAULT_ENV_CONFIGS
        metafunc.parametrize("env_dispatch", env_configs, indirect=True, scope="class")
