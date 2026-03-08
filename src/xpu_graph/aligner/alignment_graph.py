from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


class Stage(IntEnum):
    FORWARD = 0
    BACKWARD = 1


@dataclass
class AlignmentNode:
    id: int
    stage: Stage
    meta: dict[str, Any] = field(default_factory=dict)
    data: dict[str, list] = field(default_factory=dict)  # variant_id → list of (xorsum, tensor, ...) pairs

    def record(self, variant_id: str, *args):
        self.data.setdefault(variant_id, []).append(args)

@dataclass
class AlignmentEdge:
    src_id: int
    dst_id: int
    ops: list[str] = field(default_factory=list)


class AlignmentGraph:
    """Dependency graph of alignment anchors for **one** Module class.

    The *same* ``AlignmentGraph`` is shared across every variant of the same
    module: eager vs compile, different precisions (fp16/bf16/fp32), different
    optimization levels, different devices.  Each variant deposits its captured
    tensor data into the graph's ``AlignmentNode`` instances under a unique
    *variant_id*.

    For **different** module definitions (e.g. a bigop ``autograd.Function`` vs
    an ``nn.Module``, an ``nn.Module`` vs. another homogeneous ``nn.Module``), 
    separate ``AlignmentGraph`` s are created and related via ``GraphMapping``.
    """

    def __init__(self, id: str):
        self.id = id
        self._nodes: dict[int, AlignmentNode] = {}
        self._topology:dict[int, list[int]] = {}    # src_id → {dst_ids}
        self._grad_links: dict[int, int] = {}       # fw_node_id → bw_node_id
        self._variant_ids: set[str] = set()

    @property
    def nodes(self) -> dict[int, AlignmentNode]:
        return self._nodes
    @property
    def topology(self) -> dict[int, list[int]]:
        return self._topology
    @property
    def grad_links(self) -> dict[int, int]:
        return self._grad_links
    @property
    def variant_ids(self) -> list[str]:
        return list(self._variant_ids)
    
    def get_node(self, node_id: int, stage: Stage = Stage.FORWARD) -> AlignmentNode:
        if node_id not in self._nodes:
            self._nodes[node_id] = AlignmentNode(id=node_id, stage=stage)
        return self._nodes[node_id]

    def add_edge(self, src_id: int, dst_id: int):
        self._topology.setdefault(src_id, list()).append(dst_id)

    def add_grad_link(self, fw_node_id: int, bw_node_id: int):
        self._grad_links[fw_node_id] = bw_node_id

    def register_variant(self, variant_id: str):
        if variant_id not in self._variant_ids:
            self._variant_ids.add(variant_id)

    def remove_variant(self, variant_id: str):
        self._variant_ids.remove(variant_id)
        for node in self._nodes.values():
            node.data.pop(variant_id, None)
    
    def clear_data(self):
        for node in self._nodes.values():
            node.data.clear()

    def __repr__(self):
        return (
            f"AlignmentGraph(id={self.id!r}, "
            f"nodes={len(self._nodes)}, "
            f"edges={sum(len(v) for v in self._topology.values())}, "
            f"variants={self._variant_ids})"
        )


class GraphMapping:
    """Bidirectional node mapping between two structurally different ``AlignmentGraph``s.

    Used when comparing models written in different paradigms (bigop
    ``autograd.Function`` vs ``nn.Module``) that represent the same logical
    computation.  The two graphs may have completely different topologies, but
    their semantically corresponding alignment anchors are linked here.
    """

    pass