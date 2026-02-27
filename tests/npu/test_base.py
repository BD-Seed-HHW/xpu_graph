import pytest
import torch
from xpu_graph.compiler import Target, XpuGraph, XpuGraphConfig


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.ops.aten._to_copy(self.linear(x).relu(), dtype=torch.int32)


class TestGeAndAclGraphMode:
    def setup_method(self):
        torch.npu.set_device(0)
        self.module = MyModule().eval().npu()

        # WARNING(liuyuan): It is necessary to provide vendor_compiler_config to enalbe GE.
        self.ge_func = torch.compile(
            self.module,
            backend=XpuGraph(
                XpuGraphConfig(
                    False,
                    target=Target.npu,
                    freeze=True,  # WARNING(liuyuan): Critical for nn.Module with Parameter under pytorch 2.5-
                    debug=True,
                    vendor_compiler_config={"compiler": "ge"},
                )
            ),
            dynamic=False,
            fullgraph=True,
        )
        assert self.ge_func is not None
        self.acl_graph_func = torch.compile(
            self.module,
            backend=XpuGraph(
                XpuGraphConfig(
                    False,
                    target=Target.npu,
                    freeze=True,  # WARNING(liuyuan): Critical for nn.Module with Parameter under pytorch 2.5-
                    debug=True,
                    vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
                )
            ),
            dynamic=False,
            fullgraph=True,
        )
        assert self.acl_graph_func is not None

        self.compiled_with_default_cfg = torch.compile(
            self.module,
            backend=XpuGraph(
                XpuGraphConfig(
                    False,
                    target=Target.npu,
                    freeze=True,  # WARNING(liuyuan): Critical for nn.Module with Parameter under pytorch 2.5-
                    debug=True,
                    vendor_compiler_config={},  # NOTE(liuyuan): use default vendor config, which is the same as acl_graph one above.
                )
            ),
            dynamic=False,
            fullgraph=True,
        )

    # WARNING(liuyuan): ACL Graph does not support variable and dynamic shape.
    @pytest.mark.parametrize("shape", [(32,)])
    def testInference(self, shape):
        input = torch.randn((*shape, 4)).npu()
        torch.testing.assert_close(self.module(input), self.ge_func(input), rtol=1e-03, atol=1e-03, equal_nan=True)
        torch.testing.assert_close(
            self.module(input), self.acl_graph_func(input), rtol=1e-03, atol=1e-03, equal_nan=True
        )
        torch.testing.assert_close(
            self.ge_func(input), self.acl_graph_func(input), rtol=1e-03, atol=1e-03, equal_nan=True
        )
        torch.testing.assert_close(
            self.compiled_with_default_cfg(input), self.acl_graph_func(input), rtol=1e-03, atol=1e-03, equal_nan=True
        )


if __name__ == "__main__":
    testObj = TestGeAndAclGraphMode()
    testObj.setup_method()
    testObj.testInference((32,))
