import re
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer

from .bucketing_utils import bucket_key, is_wait_tensor, merge_all_gather_bucket, merge_reduce_scatter_bucket
from .bucketing_utils import is_all_gather_into_tensor as is_all_gather
from .bucketing_utils import is_reduce_scatter_tensor as is_reduce_scatter
from .reordering import _stable_topological_sort

#  adapted from https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/overlap_manual_scheduling.py

@dataclass
class CollectiveInfo:
    start_node: fx.Node
    wait_node: fx.Node


class COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_TO_ALL = 3
    UNSUPPORTED = 4


def _schedulable_wait_node(node: torch.fx.Node) -> bool:
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


class ManualBucketer:
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
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.graph.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def _collect_node_users(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
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
            merge_all_gather_bucket(
                self.graph,
                coll_nodes,
                wait_insertion_point=first_wait,
                insert_before=next_node,
            )
        elif is_reduce_scatter(first):
            merge_reduce_scatter_bucket(
                self.graph,
                coll_nodes,
                wait_insertion_point=first_wait,
                insert_before=next_node,
            )
        else:
            raise ValueError(
                "bucket non all_gather/reduce_scatter node is not supported"
            )

    def manual_bucket_collectives(self, nodes: list[fx.Node]) -> None:
        # Filter out valid collectives
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives == []:
            return
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in collectives:
            key = bucket_key(node, self.bucket_mode)
            if not (is_all_gather(node) or is_reduce_scatter(node)):
                continue
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
        for key, nodes in grouped_collectives.items():
            self._bucket_group(list(nodes))


class Bucketing(Optimizer):

    _support_stages = [
        FxStage.inference,
        FxStage.forward,
        FxStage.backward,
    ]

    def __init__(self, module_bucket_plans: list[list[str] | str]):
        self.module_bucket_plans = module_bucket_plans

    def process(self, gm: fx.GraphModule):
        gm.graph.lint()
        for node in gm.graph.nodes:
            if node.op == "call_function" and (node.target == torch.ops.bucketing._pre_bucket_all_gather or node.target == torch.ops.bucketing._pre_bucket_reduce_scatter):
                return False

        self.graph = gm.graph
        self.collective_info: dict[fx.Node, CollectiveInfo] = {}
        self._identify_collectives()
        self.node_idx = {n: i for i, n in enumerate(self.graph.nodes)}
        self.bucketer = ManualBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_idx=self.node_idx,
            insert_overlap_deps=True,
        )

        self._manual_bucket_collectives()
        return True

    def _identify_collectives(self) -> None:
        for node in self.graph.nodes:
            if _schedulable_wait_node(node):
                start = node.args[0]
                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                )
                self.collective_info[start] = info

    def _manual_bucket_collectives(self) -> None:
        self._get_nodes_in_plans()
        for nodes in self.nodes_in_plans:
            self.bucketer.manual_bucket_collectives(nodes=nodes)
        _stable_topological_sort(self.graph, {})
        self.graph.lint()

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
