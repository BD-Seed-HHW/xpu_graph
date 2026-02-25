import pytest
import torch
from xpu_graph import Target, XpuGraph, XpuGraphConfig


class BMM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2048, 1024))

    def forward(self, x):
        return torch.matmul(x, self.weight)


@pytest.mark.skip(reason="toxic, see #396")
class TestMemPoolReuse:
    def setup_method(self):
        self.model = BMM().npu()
        # NOTE(liuyuan): heuristically that we found that we will have 0 memory growth under such buckets.
        self.input_list = [torch.randn(2**i, 2048, 2048).npu() for i in range(8, 1, -1)]
        # self.input_list = [
        #     torch.randn(2**i, 3, 128, 128).npu() for i in range(5, 1, -1)
        # ]
        torch.npu.config.allow_internal_format = False

    def test_respective_mem_pool(self):
        compiler = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                target=Target.npu,
                vendor_compiler_config={
                    "compiler": "ge",
                    "mode": "reduce-overhead",
                },
            )
        )

        with torch.inference_mode():
            compiled_model = torch.compile(self.model, backend=compiler, dynamic=False)

            for input_tensor in self.input_list:
                memory_reserved = torch.npu.max_memory_reserved()
                compiled_model(input_tensor)
                # NOTE(liuyuan): Memory allocation should keep growing.
                assert memory_reserved < torch.npu.max_memory_reserved()

    def test_mem_pool_reuse(self):
        mem_pool = torch.npu.graph_pool_handle()
        compiler = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                target=Target.npu,
                vendor_compiler_config={
                    "compiler": "ge",
                    "mode": "reduce-overhead",
                    "use_custom_pool": mem_pool,
                },
            )
        )

        with torch.inference_mode():
            compiled_model = torch.compile(self.model, backend=compiler, dynamic=False)

            # NOTE(liuyuan): do the first graph capture and memory allocation.
            compiled_model(self.input_list[0])
            memory_reserved = torch.npu.max_memory_reserved()

            mem_growth = []
            for input_tensor in self.input_list[1:]:
                compiled_model(input_tensor)
                # assert memory_reserved == torch.npu.max_memory_reserved()
                mem_growth.append(torch.npu.max_memory_reserved() - memory_reserved)
                memory_reserved = torch.npu.max_memory_reserved()

            # NOTE(liuyuan): should grow with 0B at least once if we use acl_graph memory pool reuse.
            assert 0 in mem_growth, f"{mem_growth=}"
