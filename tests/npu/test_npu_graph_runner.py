import pytest
import torch
from xpu_graph.compiler import Target, XpuGraph, XpuGraphConfig


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.ops.aten._to_copy(self.linear(x).relu(), dtype=torch.int32)


class TestDeviceGraphCompiler:
    def setup_method(self):
        torch.npu.set_device(0)
        self.module = MyModule().eval().npu()

        self.device_graph_func = torch.compile(
            self.module,
            backend=XpuGraph(
                XpuGraphConfig(
                    False,
                    target=Target.npu,
                    freeze=True,  # WARNING(liuyuan): Critical for nn.Module with Parameter under pytorch 2.5-
                    debug=True,
                    vendor_compiler_config={"compiler": "device_graph"},
                )
            ),
            dynamic=False,
            fullgraph=False,  # NOTE: allow graph breaks; Device Graph capture/replay may not support fullgraph for all models yet
        )
        assert self.device_graph_func is not None

    @pytest.mark.parametrize("shape", [(32,)])
    def testInference(self, shape):
        input = torch.randn((*shape, 4)).npu()
        torch.testing.assert_close(
            self.module(input), self.device_graph_func(input), rtol=1e-03, atol=1e-03, equal_nan=True
        )
        input2 = torch.randn((*shape, 4)).npu()
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
                # Part 1: NPU (Supported)
                x = self.l1(x)

                # Part 2: CPU op (Unsupported in NPU Graph context)
                # Moving to CPU breaks the device consistency check in DeviceGraphSupport
                x = x.cpu()
                x = x + 1.0
                x = x.npu()

                # Part 3: NPU (Supported)
                x = self.l2(x)
                return x

        model = MixedModel().npu()
        input_tensor = torch.randn(16, 32).npu()

        # 1. Capture FX Graph
        gm = make_fx(model)(input_tensor)

        # 2. Compile with partitioning
        # This will roughly split into: [NPU Submodule] -> [CPU Ops in Main Graph] -> [NPU Submodule]
        compiled_gm = device_graph_compiler(gm, [input_tensor], "npu")

        # 3. Execution (Capture & Replay 1)
        res1 = compiled_gm(input_tensor)
        golden = model(input_tensor)
        assert torch.allclose(res1, golden, atol=1e-3)

        # 4. Replay 2 (with new input values)
        input_tensor_2 = torch.randn(16, 32).npu()
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
    testObj = TestDeviceGraphCompiler()
    testObj.setup_method()
    testObj.testInference((32,))
