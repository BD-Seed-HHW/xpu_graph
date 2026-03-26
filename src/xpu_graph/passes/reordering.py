from collections import defaultdict

import torch
import torch.fx as fx
from torch.fx import Node, map_arg
from torch.utils._ordered_set import OrderedSet

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer

from .bucketing_utils import is_all_gather_into_tensor, is_reduce_scatter_tensor

#  adapted from https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/overlap_manual_scheduling.py


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _stable_topological_sort(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
    do_sort: bool = True,
) -> bool:
    pending = list(reversed(graph.nodes))
    ready = OrderedSet[Node]()
    waiting = defaultdict(list)
    outputs = OrderedSet[Node]()
    cursor = None
    while pending:
        node = pending.pop()

        if node.target == "output":
            outputs.add(node)
            assert not node.users, "output nodes should have no users"
            continue

        waiting_for = [
            x
            for x in _get_flat_args_unique(node, node_to_additional_deps)
            if x not in ready
        ]
        if waiting_for:
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node and do_sort:
                cursor.append(node)
            cursor = node
            pending.extend(reversed(waiting.pop(node, ())))

    ready.update(outputs)
    return not waiting and len(ready) == len(graph.nodes)


class Reordering(Optimizer):

    _support_stages = [
        FxStage.inference,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        gm.graph.lint()
        self.gm = gm
        self.graph = gm.graph
        if torch.distributed.get_rank() == 0:
            print(f"before reordering, graph :\n{gm.print_readable()}")
        for node in self.graph.nodes:
            if node.meta.get("manual_reorder_node_type") == "rs" or node.meta.get("manual_reorder_node_type") == "ag":
                return False
        self.manual_reorder_graph()
        if torch.distributed.get_rank() == 0:
            print(f"after reordering, graph :\n{gm.print_readable()}")
        return True

    def manual_reorder_graph(self) -> None:
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        last_rs_wait_node: fx.Node = None
        for node in self.graph.nodes:
            if is_reduce_scatter_tensor(node):
                if last_rs_wait_node is not None:
                    overlap_deps[last_rs_wait_node].add(node)
                    node.meta["manual_reorder_node_type"] = "rs"
                last_rs_wait_node = next(iter(node.users))
        last_ag_wait_node: fx.Node = None
        for node in self.graph.nodes:
            if is_all_gather_into_tensor(node):
                if last_ag_wait_node is not None:
                    overlap_deps[last_ag_wait_node].add(node)
                    node.meta["manual_reorder_node_type"] = "ag"
                last_ag_wait_node = next(iter(node.users))
        _stable_topological_sort(self.graph, overlap_deps, do_sort=True)
