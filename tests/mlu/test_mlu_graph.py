import pytest
import torch
from xpu_graph.compiler import XpuGraph, XpuGraphConfig
from xpu_graph.config import Target
from xpu_graph.device_graph_runner import GraphRunner


class TestMluGraphRunner:
    def test_mlu_graph_runner_basic(self):
        model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024)] * 3).mlu()
        input_tensor = torch.empty(1024, 1024).uniform_(10, 100).mlu()

        golden = model(input_tensor)
        device_graph = GraphRunner[Target.mlu](
            model,
            lambda input_tensor: input_tensor,
            lambda input_buffer, input_tensor: (input_buffer.copy_(input_tensor), True),
        )
        device_graph.capture(torch.empty_like(input_tensor).uniform_(1, 2))

        assert torch.allclose(device_graph(input_tensor), golden)

        device_graph_2 = device_graph.clone()
        device_graph_2.capture(torch.empty_like(input_tensor).uniform_(3, 4), clone_args=True)

        assert torch.allclose(device_graph_2(input_tensor), golden)


class TestDeviceGraphCompiler:
    def setup_method(self):
        self.module = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024)] * 3).mlu()

        self.device_graph_func = torch.compile(
            self.module,
            backend=XpuGraph(
                XpuGraphConfig(
                    False,
                    target=Target.mlu,
                    freeze=True,  # WARNING(liuyuan): Critical for nn.Module with Parameter under pytorch 2.5-
                    debug=True,
                    vendor_compiler_config={"compiler": "device_graph"},
                )
            ),
            dynamic=False,
            fullgraph=False,  # NOTE: allow graph breaks; Device Graph capture/replay may not support fullgraph for all models yet
        )
        assert self.device_graph_func is not None

    @pytest.mark.parametrize("shape", [(1024, 1024)])
    def testInference(self, shape):
        input1 = torch.randn(*shape).mlu()
        torch.testing.assert_close(
            self.module(input1), self.device_graph_func(input1), rtol=1e-03, atol=1e-03, equal_nan=True
        )
        input2 = torch.randn(*shape).mlu()
        torch.testing.assert_close(
            self.module(input2), self.device_graph_func(input2), rtol=1e-03, atol=1e-03, equal_nan=True
        )

    def test_compiler_partitioning_mixed_ops(self):
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx
        from xpu_graph.backends.device_graph import device_graph_compiler

        class MixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(32, 32)
                self.l2 = torch.nn.Linear(32, 32)

            def forward(self, x):
                # Part 1: MLU (Supported)
                x = self.l1(x)

                # Part 2: CPU op (Unsupported in MLU Graph context)
                # Moving to CPU breaks the device consistency check in DeviceGraphSupport
                x = x.cpu()
                x = x + 1.0
                x = x.mlu()

                # Part 3: MLU (Supported)
                x = self.l2(x)
                return x

        model = MixedModel().mlu()
        input_tensor = torch.randn(16, 32).mlu()

        # 1. Capture FX Graph
        gm = make_fx(model)(input_tensor)

        # 2. Compile with partitioning
        # This will roughly split into: [MLU Submodule] -> [CPU Ops in Main Graph] -> [MLU Submodule]
        compiled_gm = device_graph_compiler(gm, [input_tensor], "mlu")

        # 3. Execution (Capture & Replay 1)
        res1 = compiled_gm(input_tensor)
        golden = model(input_tensor)
        assert torch.allclose(res1, golden, atol=1e-3)

        # 4. Replay 2 (with new input values)
        input_tensor_2 = torch.randn(16, 32).mlu()
        res2 = compiled_gm(input_tensor_2)
        golden_2 = model(input_tensor_2)
        assert torch.allclose(res2, golden_2, atol=1e-3)

        # 5. Verify Partitioning
        # We expect _LazyXPUGraph wrappers to be present for the supported parts
        lazy_runner_count = 0
        for _, submod in compiled_gm.named_children():
            if submod.__class__.__name__ == "_LazyXPUGraph":
                lazy_runner_count += 1

        # Typically 2 partitions: before and after the CPU op
        assert lazy_runner_count >= 1, "Should have created at least one partitioned subgraph"


if __name__ == "__main__":
    runnerTest = TestMluGraphRunner()
    runnerTest.test_mlu_graph_runner_basic()

    testObj = TestDeviceGraphCompiler()
    testObj.setup_method()
    testObj.testInference((1024, 1024))
    testObj.test_compiler_partitioning_mixed_ops()
