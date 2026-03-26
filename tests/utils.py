import pytest


def parametrize_class_env(env_list):
    return pytest.mark.parametrize("class_env_setup", env_list, indirect=True)
