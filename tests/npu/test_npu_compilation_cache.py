import pytest
import torch
from xpu_graph import Target
from xpu_graph.backends.npu import NpuSerializableArtifact
from xpu_graph.cache import XpuGraphLocalCache
from xpu_graph.compiler import XpuGraphCompiler
from xpu_graph.fx_utils import FxStage
from xpu_graph.test_utils import need_xpu_graph_logs


@pytest.mark.exclusive
class TestNpuCompilationCache:
    def setup_method(self):
        self.compiler = XpuGraphCompiler()
        self.inputs = (torch.empty(1024).npu(), torch.empty(1024).npu())
        self.stage = FxStage.inference

    @pytest.mark.parametrize(
        "compiler_setting",
        (
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge", "mode": "reduce-overhead"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(False)
            .done(),
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(False)
            .done(),
        ),
        ids=("aclgraph", "ge"),
    )
    def test_cache(self, tmp_path, compiler_setting):
        cache = XpuGraphLocalCache(tmp_path)
        compiler_setting(self)
        if not self.compiler._compiler._config.fallback_legacy_dispatch:
            pytest.skip("This test requires legacy_dispatch which returns fresh artifact")
        ori_gm = []
        compiled_gm = []

        def dummy(x, y):
            return x + y

        def hijack_backend(gm, *args, **kwargs):
            ori_gm.append(gm)
            compiled_gm.append(self.compiler._compiler(gm, *args, **kwargs))
            return compiled_gm[0]

        compiled_module = torch.compile(dummy, backend=hijack_backend, dynamic=False)
        compiled_module(*self.inputs)

        cache_key = cache.cache_key(
            ori_gm[0],
            self.inputs,
            self.compiler._config,
            self.stage,
        )

        cache.save_artifact(cache_key, NpuSerializableArtifact(compiled_gm[0]._compiled_func))
        compiled_module = cache.load_artifact(cache_key).artifact
        assert torch.allclose(compiled_module(*self.inputs)[0], dummy(*self.inputs), equal_nan=True)

    @pytest.mark.parametrize(
        "compiler_setting",
        (
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge", "mode": "reduce-overhead"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(True)
            .done(),
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(True)
            .done(),
        ),
        ids=("aclgraph", "ge"),
    )
    def test_compile_pipeline(self, compiler_setting, caplog):
        compiler_setting(self)

        class Model(torch.nn.Module):
            def __init__(self, const_0, const_1):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(1024))
                self.const_0 = const_0
                self.const_1 = const_1

            def forward(self, x, y):
                return (x + y + self.weight) * self.const_0 * self.const_1

        with need_xpu_graph_logs():
            model = Model(1, 4).eval().npu()
            compiled = self.compiler.compile(model, dynamic=False)
            with torch.no_grad():
                compiled(*self.inputs)

            torch._dynamo.reset()
            compiled = self.compiler.compile(model, dynamic=False)

            with torch.no_grad():
                assert torch.allclose(compiled(*self.inputs), model(*self.inputs), equal_nan=True)
            assert "Use cache in location" in caplog.text

            # torch._dynamo.reset()
            model = Model(2, 6).eval().npu()
            compiled = self.compiler.compile(model, dynamic=False)
            with torch.no_grad():
                assert torch.allclose(compiled(*self.inputs), model(*self.inputs), equal_nan=True)

            # torch._dynamo.reset()
            model = Model(torch.ones(1, dtype=torch.int32).npu(), torch.ones(1, dtype=torch.int32).npu()).eval().npu()
            compiled = self.compiler.compile(model, dynamic=False)
            with torch.no_grad():
                assert torch.allclose(compiled(*self.inputs), model(*self.inputs), equal_nan=True)

            # FIXME(liuyuan): However, the results of constant folding passes, which are not applied to cpu variables,  will not save by npu backend, fix it.
