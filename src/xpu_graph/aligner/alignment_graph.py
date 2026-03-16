from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch


class Stage(IntEnum):
    FORWARD = 0
    BACKWARD = 1


@dataclass(frozen=True)
class OpInfo:
    op: str
    target: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    name: str | None = None

    @property
    def signature(self) -> tuple[Any, ...]:
        return (self.op, self.target, self.args, self.kwargs)

    @classmethod
    def from_fx_node(cls, fxnode: torch.fx.Node) -> "OpInfo":
        return cls(
            op=fxnode.op,
            target=cls.normalize_target(fxnode.target),
            args=tuple(cls.normalize_value(arg) for arg in fxnode.args),
            kwargs=tuple(sorted((str(key), cls.normalize_value(value)) for key, value in fxnode.kwargs.items())),
            name=fxnode.name,
        )

    @staticmethod
    def normalize_target(target: Any) -> str:
        if isinstance(target, str):
            return target
        objclass = getattr(target, "__objclass__", None)
        name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
        if objclass is not None and name:
            class_module = getattr(objclass, "__module__", None)
            class_name = getattr(objclass, "__qualname__", None) or getattr(objclass, "__name__", None)
            if class_module and class_name:
                return f"{class_module}.{class_name}.{name}"
        module = getattr(target, "__module__", None)
        if module and name:
            return f"{module}.{name}"
        overloadpacket = getattr(target, "overloadpacket", None)
        if overloadpacket is not None:
            return str(overloadpacket)
        return str(target)

    @classmethod
    def normalize_value(cls, value: Any):
        if isinstance(value, torch.fx.Node):
            return ("fx_node", value.op)
        if isinstance(value, slice):
            return ("slice", value.start, value.stop, value.step)
        if isinstance(value, (list, tuple)):
            return tuple(cls.normalize_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), cls.normalize_value(v)) for k, v in value.items()))
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
            return str(value)
        return cls.normalize_target(value)


@dataclass
class AlignmentNode:
    id: int
    stage: Stage
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)  # field -> variant_id -> value
    data: dict[str, list] = field(default_factory=dict)  # variant_id -> list of (xorsum, tensor, ...) pairs

    def record_data(self, variant_id: str, *args):
        self.data.setdefault(variant_id, []).append(args)

    def set_meta(self, field: str, variant_id: str, value: Any):
        self.meta.setdefault(field, {})[variant_id] = value

    def get_meta(self, field: str, variant_id: str, default: Any = None) -> Any:
        return self.meta.get(field, {}).get(variant_id, default)


@dataclass
class AlignmentEdge:
    """Represents a directed edge between two ``AlignmentNode``s, recording ops executed between them."""
    id: int
    src_id: int
    dst_id: int
    ops: tuple[OpInfo] = ()
    variant_ids: set[str] = field(default_factory=set)


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
        self._edges: dict[int, AlignmentEdge] = {}
        self._topology: dict[int, list[int]] = {}  # src_id -> edge_ids
        self._edge_lookup: dict[tuple[int, int, tuple[tuple[Any, ...], ...]], int] = {}
        self._variant_edges: dict[str, set[int]] = {}
        self._grad_links: dict[int, int] = {}  # fw_node_id -> bw_node_id
        self._variant_ids: set[str] = set()
        self._collapsed_fw_node_ids: dict[str, set[int]] = {}
        self._next_edge_id: int = 0

    @property
    def nodes(self) -> dict[int, AlignmentNode]:
        return self._nodes
    @property
    def edges(self) -> dict[int, AlignmentEdge]:
        return self._edges
    @property
    def topology(self) -> dict[int, list[int]]:
        return {src_id: list(edge_ids) for src_id, edge_ids in self._topology.items()}
    @property
    def grad_links(self) -> dict[int, int]:
        return self._grad_links
    
    def get_node(self, node_id: int, stage: Stage = Stage.FORWARD) -> AlignmentNode:
        if node_id not in self._nodes:
            self._nodes[node_id] = AlignmentNode(id=node_id, stage=stage)
        return self._nodes[node_id]

    def add_edge(self, variant_id: str, src_id: int, dst_id: int, ops: tuple[OpInfo, ...]) -> AlignmentEdge:
        edge_signature = tuple(op.signature for op in ops)
        edge_key = (src_id, dst_id, edge_signature)
        edge_id = self._edge_lookup.get(edge_key)
        if edge_id is None:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
            self._edges[edge_id] = AlignmentEdge(
                id=edge_id,
                src_id=src_id,
                dst_id=dst_id,
                ops=ops,
            )
            self._edge_lookup[edge_key] = edge_id
            self._topology.setdefault(src_id, []).append(edge_id)

        edge = self._edges[edge_id]
        edge.variant_ids.add(variant_id)
        self._variant_edges.setdefault(variant_id, set()).add(edge_id)
        return edge

    def iter_edges(self, variant_id: str | None = None):
        if variant_id is None:
            for edge_id in sorted(self._edges):
                yield self._edges[edge_id]
            return
        for edge_id in sorted(self._variant_edges.get(variant_id, set())):
            yield self._edges[edge_id]

    def get_edges_between(self, src_id: int, dst_id: int, variant_id: str | None = None) -> list[AlignmentEdge]:
        edge_ids = self._topology.get(src_id, [])
        return [
            self._edges[edge_id]
            for edge_id in edge_ids
            if self._edges[edge_id].dst_id == dst_id
            and (variant_id is None or variant_id in self._edges[edge_id].variant_ids)
        ]

    def add_grad_link(self, fw_node_id: int, bw_node_id: int):
        self._grad_links[fw_node_id] = bw_node_id

    def mark_collapsed_fw_node(self, variant_id: str, node_id: int):
        self._collapsed_fw_node_ids.setdefault(variant_id, set()).add(node_id)

    def is_fw_node_collapsed(self, variant_id: str, node_id: int) -> bool:
        return node_id in self._collapsed_fw_node_ids.get(variant_id, set())

    def register_variant(self, variant_id: str):
        if variant_id not in self._variant_ids:
            self._variant_ids.add(variant_id)
            self._variant_edges.setdefault(variant_id, set())
            self._collapsed_fw_node_ids.setdefault(variant_id, set())

    def clear_data(self):
        for node in self._nodes.values():
            node.data.clear()

    def __repr__(self):
        return (
            f"AlignmentGraph(id={self.id!r}, "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
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
