import heapq
from collections import Counter, defaultdict

import torch
import torch.fx as fx
from torch.fx import Node, map_arg
from torch.utils._ordered_set import OrderedSet

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer

from .bucketing_utils import is_all_gather_into_tensor, is_reduce_scatter_tensor, is_wait_tensor

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
    # Nodes are in exactly one of these four collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = OrderedSet[Node]()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # - `outputs` are always at the end of the graph
    outputs = OrderedSet[Node]()

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
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
        # print(f"node: {node}, waiting_for: {waiting_for}")
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node and do_sort:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    ready.update(outputs)
    return not waiting and len(ready) == len(graph.nodes)


class Reordering(Optimizer):

    _support_stages = [
        # FxStage.inference,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        gm.graph.lint()
        self.gm = gm
        self.graph = gm.graph
        self.in_degree = Counter(user for node in self.graph.nodes for user in node.users)
        if torch.distributed.get_rank() == 0:
            print(f"before reordering, graph :\n{gm.print_readable()}")
        before_nodes = list(self.graph.nodes)
        self._manual_reorder_graph()
        after_nodes = list(self.graph.nodes)
        if torch.distributed.get_rank() == 0:
            print(f"after reordering, graph :\n{gm.print_readable()}")
        return before_nodes != after_nodes

    def _schedule(self, node: fx.Node) -> None:
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, (self.node_idx[user], user))

    def _manual_reorder_graph(self) -> None:
        """
        Reorder the graph manually based on the bucketed collectives.
        """
        delayed_rs_nodes: list[fx.Node] = []
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        self.node_idx = {n: i for i, n in enumerate(self.graph.nodes)}
        self.scheduled = OrderedSet()
        self.ready: list[tuple[int, fx.Node]] = []

        for node in self.graph.nodes:
            if self.in_degree[node] == 0:
                heapq.heappush(self.ready, (self.node_idx[node], node))
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"ready: {self.ready}")
        while self.ready:
            _, node = heapq.heappop(self.ready)
            is_rs = is_reduce_scatter_tensor(node)
            is_rs_wait = is_wait_tensor(node) and is_reduce_scatter_tensor(node.args[0])
            # if rank == 0:
            #     print(f"node: {node}, is_rs: {is_rs}, is_rs_wait: {is_rs_wait}")
            if node in self.scheduled:
                continue

            if is_rs:
                for delayed in delayed_rs_nodes:
                    self._schedule(delayed)
                    overlap_deps[delayed].add(node)
                delayed_rs_nodes.clear()

            elif is_rs_wait:
                delayed_rs_nodes.append(node)
                continue
            self._schedule(node)
        for delayed in delayed_rs_nodes:
            self._schedule(delayed)

        self.scheduled = OrderedSet(reversed(list(self.scheduled)))
        picked_ag: list[fx.Node] = []

        for node in self.scheduled:
            is_ag = is_all_gather_into_tensor(node)
            is_ag_wait = is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0])
            if is_ag:
                picked_ag.append(node)
                continue

            if is_ag_wait:
                if picked_ag:
                    reversed_picked_ag = list(reversed(picked_ag))
                    for ag in reversed_picked_ag:
                        overlap_deps[node].add(ag)
                picked_ag.clear()
        print(f"overlap_deps: {overlap_deps}")
        _stable_topological_sort(self.graph, overlap_deps)
        self.graph.lint()
