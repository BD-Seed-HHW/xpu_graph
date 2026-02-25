import torch
from xpu_graph import OptLevel, Target, XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import need_xpu_graph_logs


class TestSuperKernelScope:
    def test_super_kernel_scope_in_acl_graph(self, caplog):
        with need_xpu_graph_logs():

            def dummy(x):
                return x + x + x

            xpu_graph = XpuGraph(
                XpuGraphConfig(
                    is_training=False,
                    debug=True,
                    target=Target.npu,
                    vendor_compiler_config={"compiler": "ge", "mode": "reduce-overhead"},
                )
            )
            compiled = torch.compile(dummy, backend=xpu_graph, fullgraph=True)
            compiled(torch.empty(1024, device="npu"))
            assert "Pattern.ScopedSuperKernel changed graph" not in caplog.text

    def test_super_kernel_scope_in_ge(self, caplog):
        with need_xpu_graph_logs():

            def dummy(x):
                return x + x + x

            # NOTE(liuyuan): use GE as backend so that we could enable the pass.
            xpu_graph = XpuGraph(
                XpuGraphConfig(
                    is_training=False,
                    debug=True,
                    opt_level=OptLevel.level1,
                    vendor_compiler_config={
                        "compiler": "ge",
                        "enable_super_kernel": True,
                    },
                    target=Target.npu,
                )
            )
            compiled = torch.compile(dummy, backend=xpu_graph, fullgraph=True)
            compiled(torch.empty(1024, device="npu"))
            assert "Pattern.ScopedSuperKernel changed graph" in caplog.text
