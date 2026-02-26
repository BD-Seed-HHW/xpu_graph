import pytest
import torch
import torch.fx as fx
from torch.utils.checkpoint import CheckpointPolicy
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.reshard_after_forward import ReshardAfterForward


class TestReshardAfterForward:
    def test_support_stages(self):
        optimizer = ReshardAfterForward()
        assert FxStage.joint in optimizer._support_stages
        assert len(optimizer._support_stages) == 1

    def test_process_no_fsdp_nodes(self):
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        mod = SimpleModule()
        gm = fx.symbolic_trace(mod)

        optimizer = ReshardAfterForward()
        changed = optimizer.process(gm)

        assert changed is False

    def test_process_with_fsdp_pattern(self):
        graph = fx.Graph()
        input_node = graph.placeholder("param")

        all_gather_node = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            (input_node, 1, "default")
        )

        wait_tensor_node = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            (all_gather_node,)
        )

        graph.output(wait_tensor_node)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        optimizer = ReshardAfterForward()
        changed = optimizer.process(gm)

        assert changed is True
        assert all_gather_node.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert all_gather_node.meta["ac_graph_id"] == 100000
        assert wait_tensor_node.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert wait_tensor_node.meta["ac_graph_id"] == 100000

    def test_process_with_slice_after_wait(self):
        graph = fx.Graph()
        input_node = graph.placeholder("param")

        all_gather_node = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            (input_node, 1, "default")
        )

        wait_tensor_node = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            (all_gather_node,)
        )

        slice_node = graph.call_function(
            torch.ops.aten.slice.Tensor,
            (wait_tensor_node, 0, 0, 10)
        )

        graph.output(slice_node)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        optimizer = ReshardAfterForward()
        changed = optimizer.process(gm)

        assert changed is True
        assert slice_node.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert slice_node.meta["ac_graph_id"] == 100000

    def test_process_with_dtype_cast_before_all_gather(self):
        graph = fx.Graph()
        input_node = graph.placeholder("param")

        convert_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            (input_node, torch.float32)
        )

        all_gather_node = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            (convert_node, 1, "default")
        )

        wait_tensor_node = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            (all_gather_node,)
        )

        graph.output(wait_tensor_node)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        optimizer = ReshardAfterForward()
        changed = optimizer.process(gm)

        assert changed is True
        assert convert_node.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert convert_node.meta["ac_graph_id"] == 100000

    def test_get_pass_with_stage(self):
        optimizer = ReshardAfterForward()

        assert optimizer.get_pass_with_stage(FxStage.joint) == optimizer
        assert optimizer.get_pass_with_stage(FxStage.inference) is None
        assert optimizer.get_pass_with_stage(FxStage.forward) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
