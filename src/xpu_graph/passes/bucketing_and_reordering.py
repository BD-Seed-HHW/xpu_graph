from dataclasses import dataclass, field
from enum import IntEnum
import re
from typing import Callable

import torch
from torch.utils._ordered_set import OrderedSet

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer

import heapq
from collections import Counter, defaultdict

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet
from .bucketing import merge_all_gather_bucket, merge_reduce_scatter_bucket
from .bucketing import (
    is_wait_tensor,
    bucket_key,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter
)
from torch.fx import map_arg, Node


@dataclass
class CollectiveInfo:
    """Track info about a collective operation"""

    start_node: fx.Node
    wait_node: fx.Node
    size_bytes: int
    estimated_time_ms: float
    exposed_time_ms: float  # How much of this collective is still exposed
    hiding_nodes: OrderedSet[fx.Node] = field(default_factory=OrderedSet)

    @property
    def is_exposed(self) -> bool:
        return self.exposed_time_ms != 0


class COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_TO_ALL = 3
    UNSUPPORTED = 4


def _schedulable_wait_node(node: torch.fx.Node) -> bool:
    """
    Add additional check on if the wait node is schedulable
    We should not schedule a fx node that is:
        1. wait on a collective that is not callable
        2. wait on a non-NCCL communication node
    """
    if not is_wait_tensor(node):
        return False
    assert isinstance(node.args[0], torch.fx.Node)
    if not isinstance(node.args[0].target, Callable):
        return False
    is_callable: bool = node.args[0].op == "call_function"
    coll = None
    if "all_reduce" in node.args[0].target.name():
        coll = COLL.ALL_REDUCE
    elif "all_gather" in node.args[0].target.name():
        coll = COLL.ALL_GATHER
    elif "reduce_scatter" in node.args[0].target.name():
        coll = COLL.REDUCE_SCATTER
    elif any(comm in node.args[0].target.name() for comm in ("all_to_all", "alltoall")):
        coll = COLL.ALL_TO_ALL
    else:
        coll = COLL.UNSUPPORTED
    is_collective: bool = coll != COLL.UNSUPPORTED
    return is_callable and is_collective


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _stable_topological_sort_impl(
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


def _stable_topological_sort(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> None:
    assert _stable_topological_sort_impl(graph, node_to_additional_deps)


class ManualOverlapBucketer:
    """
    Buckets collective operations based on user specifications.
    The actual bucket happens in bucket_collectives, where all-gathers/reduce-scatters in
        `nodes` will be buckted one single all-gather/reduce-scatter.
    """

    def __init__(
        self,
        graph: fx.Graph,
        collective_info: dict[fx.Node, CollectiveInfo],
        node_idx: dict[fx.Node, int],
        insert_overlap_deps: bool = False,
        bucket_mode: str = "custom_ops_multidtype",
    ):
        self.graph = graph
        self.collective_info = collective_info
        self.node_idx = node_idx
        self.insert_overlap_deps = insert_overlap_deps
        self.bucket_mode = bucket_mode
        self.node_ancestors = self._collect_node_ancestors()
        self.node_users = self._collect_node_users()
        self.wait_to_node_map: dict[fx.Node, fx.Node] = defaultdict()

    def _collect_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.graph.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def _collect_node_users(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all users for each node."""
        node_users: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.graph.nodes:
            for output_node in list(node.users.keys()):
                node_users[node].add(output_node)
                node_users[node] |= node_users[output_node]
        return node_users

    def _check_recursive_dep(
        self,
        node: fx.Node,
        target_op: str,
        dep_dict: dict[torch.fx.Node, OrderedSet[torch.fx.Node]],
    ) -> bool:
        """
        Check if the node is directly used for fetch parameters/gradients

        TODO (ruisizhang123): currently, we assume the node only pre-fetch/update one parameter/gradient
            We should handle multiple parameters/gradients update case by checking if there are non closure
            computes along the path from primal/output to coll_node
        """
        deps: OrderedSet[fx.Node] = dep_dict[node]
        seen_target_op = 0
        for d in deps:
            if d.op == target_op:
                seen_target_op += 1

        return seen_target_op == 1

    def _bucket_group(self, coll_nodes: list[fx.Node]) -> None:
        assert len(coll_nodes) > 0, "bucketed coll_nodes should have nonzero node"

        waits = [self.collective_info[n].wait_node for n in coll_nodes]
        # Use earliest wait insertion point
        first_wait = min(waits, key=lambda w: self.node_idx[w])
        # Find insertion location
        first = coll_nodes[0]
        next_node = first
        while next_node in coll_nodes:
            next_node = next_node.next
        if is_all_gather(first):
            # default bucket mode: custom_ops, just pack some ops before all_gather to a new custom_op
            new_nodes, replacements = merge_all_gather_bucket(
                self.graph,
                coll_nodes,
                wait_insertion_point=first_wait,
                insert_before=next_node,
            )
        elif is_reduce_scatter(first):
            # default bucket mode: custom_ops, just pack some ops before reduce_scatter to a new custom_op
            new_nodes, replacements = merge_reduce_scatter_bucket(
                self.graph,
                coll_nodes,
                wait_insertion_point=first_wait,
                insert_before=next_node,
            )
        else:
            raise ValueError(
                "bucket non all_gather/reduce_scatter node is not supported"
            )
        # Identify the new wait and star
        new_waits = [n for n in new_nodes if _schedulable_wait_node(n)]
        assert len(new_waits) == 1, f"Expected exactly one new wait, got {new_waits}"
        new_wait = new_waits[0]
        new_start = new_wait.args[0]
        for n in new_nodes:
            if n == new_wait:
                node_type = node_type + "_wait"
            n.meta["manual_bucket_node_type"] = node_type
            if "wait" in node_type:
                self.wait_to_node_map[n] = new_wait

    def manual_bucket_collectives(self, nodes: list[fx.Node]) -> None:
        """
        Bucket all all-gather/reduce-scatter nodes from nodes into one all-gather/reduce-scatter.
        """
        # Filter out valid collectives
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives == []:
            return
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in collectives:
            key = bucket_key(node, self.bucket_mode)
            if not (is_all_gather(node) or is_reduce_scatter(node)):
                continue
            # We only want to bucket all-gather/reduce-scatter that
            # 1. all_gather that have ancestors dependent only on input placeholder(parameters)
            # 2. reduce scatter that the wait user node is returned as output(gradients)
            if is_all_gather(node) and not self._check_recursive_dep(
                node, "placeholder", self.node_ancestors
            ):
                continue
            if is_reduce_scatter(node) and not self._check_recursive_dep(
                self.collective_info[node].wait_node, "output", self.node_users
            ):
                continue
            if key is not None:
                grouped_collectives[key].add(node)
        for key, nodes in grouped_collectives.items():  # type: ignore[arg-type]
            self._bucket_group(list(nodes))


class BucketingAndReordering(Optimizer):

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def __init__(self, module_bucket_plans: list[list[str] | str]):
        self.module_bucket_plans = module_bucket_plans

    
    def process(self, gm: fx.GraphModule):
        assert gm.graph.lint(), "Graph is not in topological order. Please ensure the graph is properly constructed."
        for node in gm.graph.nodes:
            if node.op == "call_function" and (node.target == torch.ops.bucketing._pre_bucket_all_gather or node.target == torch.ops.bucketing._pre_bucket_reduce_scatter):
                return False

        self.graph = gm.graph
        self.collective_info: dict[fx.Node, CollectiveInfo] = {}
        self._identify_collectives()
        self.node_idx = {n: i for i, n in enumerate(self.graph.nodes)}
        self.bucketer = ManualOverlapBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_idx=self.node_idx,
            insert_overlap_deps=True,
        )

        self._manual_bucket_collectives()
        self._manual_reorder_graph()

        return True

    def _identify_collectives(self) -> None:
        """Identify all collective operations."""
        for node in self.graph.nodes:
            if _schedulable_wait_node(node):
                start = node.args[0]
                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=0,
                    estimated_time_ms=0,
                    exposed_time_ms=0,
                )
                self.collective_info[start] = info

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, (self.node_idx[user], user))

    def _manual_reorder_graph(self) -> None:
        """
        Reorder nodes in the FX graph to enforce manual overlap dependencies.

        Enforce:
        - all_gather_start_i depends on all_gather_wait_(i-1)
        - reduce_scatter_wait_i must happen before reduce_scatter_start_(i+1)
        """
        delayed_rs_nodes: list[fx.Node] = []
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.scheduled = OrderedSet()

        for node in self.nodes:
            if self.in_degree[node] == 0:
                heapq.heappush(self.ready, (self.node_idx[node], node))
        # schedule reduce scatter normally in self._schedule
        while self.ready:
            _, node = heapq.heappop(self.ready)
            node_type = node.meta.get("manual_bucket_node_type", "")

            if node in self.scheduled:
                continue

            if node_type == "bucketed_reduce_scatter":
                # Ensure all delayed waits execute before this reduce_scatter
                for delayed in delayed_rs_nodes:
                    self._schedule(delayed)
                    overlap_deps[delayed].add(node)
                delayed_rs_nodes.clear()

            elif node_type == "bucketed_reduce_scatter_wait":
                # Defer until next reduce_scatter
                delayed_rs_nodes.append(node)
                continue
            self._schedule(node)

        for delayed in delayed_rs_nodes:
            self._schedule(delayed)

        self.scheduled = OrderedSet(reversed(list(self.scheduled)))
        picked_ag: list[fx.Node] = []

        for node in self.scheduled:
            node_type = node.meta.get("manual_bucket_node_type", "")
            if node_type == "bucketed_all_gather":
                picked_ag.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                if picked_ag:
                    reversed_picked_ag = list(reversed(picked_ag))
                    for ag in reversed_picked_ag:
                        overlap_deps[self.bucketer.wait_to_node_map[node]].add(ag)
                picked_ag.clear()

        _stable_topological_sort(self.graph, overlap_deps)
        self.graph.lint()

    # bucketing 
    def _manual_bucket_collectives(self) -> None:
        """Bucket nodes in each module_bucket from module_bucket_plans."""
        self._get_nodes_in_plans(self.graph)
        for i, nodes in enumerate(self.nodes_in_plans):
            self.bucketer.manual_bucket_collectives(nodes=nodes)

        _stable_topological_sort(self.graph, {})
        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.graph.nodes for user in node.users)

    def _get_nodes_in_plans(self)->None:
        nodes = self.graph.nodes
        self.nodes_in_plans = [[] for _ in self.module_bucket_plans]
        for node in nodes:
            stack_name, stack_class = self.get_module_stack_from_node(node)
            if not stack_name:
                continue
            for i, plan in enumerate(self.module_bucket_plans):
                if isinstance(plan, list):
                    for module in plan:
                        if stack_name.startswith(module):
                            self.nodes_in_plans[i].append(node)
                else:
                    if stack_name.startswith(plan):
                        self.nodes_in_plans[i].append(node)

    def get_module_stack_from_node(self, node: fx.Node):
        stack_name, stack_class = None, None
        # only consider the last module in the stack
        if "nn_module_stack" in node.meta.keys():
            stack_name, stack_class = list(node.meta.get("nn_module_stack", "").values())[-1]
        elif "fwd_nn_module_stack" in node.meta.keys():
            stack_name, stack_class = list(node.meta.get("fwd_nn_module_stack", "").values())[-1]

        if stack_name and stack_class:
            cleaned = re.sub(r"^L\['self'\]\.?", "", stack_name)
            parts = re.findall(r"\['([^']+)'\]", cleaned)
            stack_name = ".".join(parts) if parts else cleaned

        return stack_name, stack_class