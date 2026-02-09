import torch
import torch.fx as fx
from torch.utils.checkpoint import CheckpointPolicy

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
        n: torch.fx.Node = node.all_input_nodes[0]
        while len(n.all_input_nodes) == 1:
            if is_graph_input(n.all_input_nodes[0]):
                return True
            n = n.all_input_nodes[0]
    return False


def force_recompute_node(node):
    node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    node.meta["ac_graph_id"] = 100000


class ReshardAfterForward(Optimizer):
    _support_stages = [
        FxStage.joint
    ]

    def process(self, gm: fx.GraphModule):
        change_graph = False
        for node in gm.graph.nodes:
            if is_wait_tensor_from_fsdp(node):
                change_graph = True
                ag_node = node.args[0]
                force_recompute_node(ag_node)  # all_gather
                force_recompute_node(node)  # wait_tensor
                # Force-recompute slice that comes after wait
                for user in node.users:
                    if (
                        user.op == "call_function"
                        and user.target == torch.ops.aten.slice.Tensor
                    ):
                        force_recompute_node(user)

                # Force-recompute potential dtype casts from all_gather
                if (
                    ag_node.all_input_nodes[0].op == "call_function"
                    and ag_node.args[0].target
                    == torch.ops.prims.convert_element_type.default
                ):
                    force_recompute_node(ag_node.all_input_nodes[0])
        return change_graph
