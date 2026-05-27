import json as _json
import math
import re
import shutil
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import graphviz
import torch

from .alignment_graph import AlignmentGraph, AlignmentNode, OpInfo, Stage


class AlignmentVisualizer:
    _VARIANT_COLORS = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
    ]
    _STAGE_COLORS = {
        Stage.FORWARD: {"stroke": "#4472C4", "fill": "#DEEBF7"},
        Stage.BACKWARD: {"stroke": "#AD683A", "fill": "#FBE5D6"},
    }

    @staticmethod
    def _node_attrs(align_node: AlignmentNode, node_id: int, gold_vid: str, variant_ids: list[str], steps: list[int]) -> dict[str, str]:
        node_payload = AlignmentVisualizer()._build_node_payload(align_node, node_id, gold_vid, variant_ids, steps)
        nbsp = "\u00A0"
        tip_lines = [f"{'-'*80}\nNode {node_id} - {node_payload['op_name']}\n{'-'*80}"]
        for group in node_payload["step_details"]:
            tip_lines.append(f"Step {group['step']}:")
            for detail in group["variants"]:
                if detail.get("status") == "no data":
                    tip_lines.append(f"  {detail['variant']}: no data")
                else:
                    flag = "✓" if detail.get("closeto", False) else "✗"
                    gold_sign = " (gold)" if detail["variant"] == gold_vid else ""
                    max_diff = detail.get("max_diff")
                    max_diff_text = "?" if max_diff is None else f"{max_diff:.9f}"
                    tip_lines.append(
                        f"  {detail['variant']+':':15} {detail.get('dtype','?'):8} "
                        f"xor={detail.get('xorsum','?'):11} Δ={max_diff_text} {flag}{gold_sign}"
                    )
        tooltip = "\n".join(tip_lines).replace(" ", nbsp)
        comment = _json.dumps(
            {
                "node_id": node_payload["node_id"],
                "stage": node_payload["stage"],
                "op_name": node_payload["op_name"],
                "steps": node_payload["step_details"],
            },
            ensure_ascii=False,
        )

        return {
            "label": node_payload["label"],
            "tooltip": tooltip,
            "comment": comment,
            "style": "filled",
            "fillcolor": node_payload["fillcolor"],
            "color": node_payload["color"],
            "fontcolor": "#333333",
        }

    @staticmethod
    def _module_stack_to_path(module_stack) -> tuple[str, ...]:
        if not module_stack:
            return ()
        if isinstance(module_stack, dict):
            return tuple(frame[0] for frame in module_stack.values())
        return tuple(frame[0] for frame in module_stack if isinstance(frame, (tuple, list)) and frame)

    @staticmethod
    def _sanitize_cluster_id(path: tuple[str, ...]) -> str:
        safe = "__".join(path) if path else "root"
        return "".join(ch if ch.isalnum() else "_" for ch in safe)

    @staticmethod
    def _dot_edge_label(ops: tuple[OpInfo, ...]) -> str:
        return "|".join(
            (
                op.target.split("aten::", 1)[1].split(".", 1)[0]
                if "aten::" in op.target
                else op.target.split("::", 1)[1].split(".", 1)[0]
                if "::" in op.target
                else op.target.rsplit(".", 1)[-1]
            )
            for op in ops
        )

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: cls._json_safe(val) for key, val in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        return value

    @staticmethod
    def _module_display_path(raw_path: str) -> str:
        display = raw_path
        display = re.sub(r"^L\['([^']+)'\]", r"\1", display)
        display = re.sub(r"\._modules\['([^']+)'\]", r".\1", display)
        display = re.sub(r"\._parameters\['([^']+)'\]", r".\1", display)
        display = re.sub(r"\._buffers\['([^']+)'\]", r".\1", display)
        display = re.sub(r"\['([^']+)'\]", r".\1", display)
        display = re.sub(r"\.+", ".", display).strip(".")
        return display or "root"

    @staticmethod
    def _module_class_name(module_cls: Any) -> str | None:
        if module_cls is None:
            return None
        return getattr(module_cls, "__qualname__", None) or getattr(module_cls, "__name__", None) or str(module_cls)

    @classmethod
    def _module_frames(cls, module_stack: Any) -> list[dict[str, str | None]]:
        if not module_stack:
            return []
        if isinstance(module_stack, dict):
            iterable = module_stack.values()
        else:
            iterable = module_stack

        frames: list[dict[str, str | None]] = []
        for frame in iterable:
            if not isinstance(frame, (tuple, list)) or not frame:
                continue
            raw_path = str(frame[0])
            if not raw_path:
                continue
            module_cls = frame[1] if len(frame) > 1 else None
            frames.append(
                {
                    "raw_path": raw_path,
                    "display_path": cls._module_display_path(raw_path),
                    "class_name": cls._module_class_name(module_cls),
                }
            )
        return frames

    @staticmethod
    def _module_id(stage: Stage, display_path: str) -> str:
        stage_name = stage.name.lower()
        canonical = display_path.strip(".") or "root"
        return f"{stage_name}::{canonical}"

    def _build_node_payload(self, align_node: AlignmentNode, node_id: int, gold_vid: str, variant_ids: list[str], steps: list[int]) -> dict:
        op_meta = align_node.get_meta("op_meta", gold_vid, {})
        opinfo = op_meta.get("last_non_trivial_op") or op_meta.get("last_op")
        op_name = opinfo.target if isinstance(opinfo, OpInfo) else "?"
        label = f"{node_id}: {op_name}"

        gold_data = align_node.data.get(gold_vid, [])
        all_step_details: list[dict[str, Any]] = []
        for step in steps:
            step_variants: list[dict[str, Any]] = []
            for vid in variant_ids:
                entry: dict[str, Any] = {"variant": vid}
                data = align_node.data.get(vid, [])
                if not data or step >= len(data):
                    entry["status"] = "no data"
                else:
                    xorsum, raw = data[step][0], data[step][1]
                    raw_32 = raw.to(torch.float32)
                    entry["dtype"] = str(raw.dtype).replace("torch.", "")
                    entry["xorsum"] = f"0x{xorsum:08X}"
                    if gold_data and step < len(gold_data):
                        g32 = gold_data[step][1].to(torch.float32)
                        r32 = raw_32.to(torch.float32)
                        entry["max_diff"] = float((r32 - g32).abs().max().item())
                        entry["closeto"] = bool(torch.allclose(r32, g32, rtol=1e-3, atol=1e-5))
                step_variants.append(entry)
            all_step_details.append({"step": step, "variants": step_variants})

        stage_colors = self._STAGE_COLORS[align_node.stage]
        return self._json_safe(
            {
                "node_id": node_id,
                "stage": align_node.stage.name,
                "op_name": op_name,
                "label": label,
                "color": stage_colors["stroke"],
                "fillcolor": stage_colors["fill"],
                "step_details": all_step_details,
                "variant_presence": {vid: bool(align_node.data.get(vid)) for vid in variant_ids},
            }
        )

    def _build_stage_payload(
        self,
        stage: Stage,
        stage_nodes: dict[int, AlignmentNode],
        edges: list[dict[str, Any]],
        gold_vid: str,
        variant_ids: list[str],
        steps: list[int],
    ) -> dict[str, Any]:
        root_module_id = self._module_id(stage, "root")
        stage_payload: dict[str, Any] = {
            "stage": stage.name,
            "root_module_id": root_module_id,
            "stage_color": self._STAGE_COLORS[stage]["stroke"],
            "modules": {},
            "nodes": [],
            "edges": edges,
        }

        stage_payload["modules"][root_module_id] = {
            "module_id": root_module_id,
            "display_name": "root",
            "display_path": "root",
            "raw_path": None,
            "class_name": None,
            "parent_module_id": None,
            "child_module_ids": set(),
            "direct_node_ids": [],
            "descendant_node_ids": [],
        }

        node_module_ids: dict[int, str] = {}

        for node_id, align_node in sorted(stage_nodes.items()):
            frames = self._module_frames(align_node.module_stack)
            parent_module_id = root_module_id
            leaf_module_id = root_module_id

            for frame in frames:
                module_id = self._module_id(stage, str(frame["display_path"]))
                module_entry = stage_payload["modules"].setdefault(
                    module_id,
                    {
                        "module_id": module_id,
                        "display_name": str(frame["display_path"]).split(".")[-1],
                        "display_path": frame["display_path"],
                        "raw_path": frame["raw_path"],
                        "class_name": frame["class_name"],
                        "parent_module_id": parent_module_id,
                        "child_module_ids": set(),
                        "direct_node_ids": [],
                        "descendant_node_ids": [],
                    },
                )
                module_entry["parent_module_id"] = parent_module_id
                stage_payload["modules"][parent_module_id]["child_module_ids"].add(module_id)
                parent_module_id = module_id
                leaf_module_id = module_id

            node_module_ids[node_id] = leaf_module_id
            stage_payload["modules"][leaf_module_id]["direct_node_ids"].append(node_id)

        def _finalize_descendants(module_id: str) -> list[int]:
            module_entry = stage_payload["modules"][module_id]
            descendants = list(sorted(module_entry["direct_node_ids"]))
            for child_id in sorted(module_entry["child_module_ids"]):
                descendants.extend(_finalize_descendants(child_id))
            module_entry["child_module_ids"] = sorted(module_entry["child_module_ids"])
            module_entry["descendant_node_ids"] = sorted(dict.fromkeys(descendants))
            return module_entry["descendant_node_ids"]

        _finalize_descendants(root_module_id)

        for node_id, align_node in sorted(stage_nodes.items()):
            node_payload = self._build_node_payload(align_node, node_id, gold_vid, variant_ids, steps)
            node_payload["module_id"] = node_module_ids[node_id]
            stage_payload["nodes"].append(node_payload)

        stage_payload["modules"] = [
            stage_payload["modules"][module_id]
            for module_id in sorted(
                stage_payload["modules"],
                key=lambda item: (item != root_module_id, stage_payload["modules"][item]["display_path"], item),
            )
        ]
        stage_payload["nodes"] = sorted(stage_payload["nodes"], key=lambda item: item["node_id"])
        stage_payload["edges"] = sorted(stage_payload["edges"], key=lambda item: item["edge_id"])
        return self._json_safe(stage_payload)

    def build_viewer_payload(
        self,
        agraph: AlignmentGraph,
        variant_ids: list[str],
        gold_vid: Optional[str] = None,
        steps: list[int] | None = None,
    ) -> dict[str, Any]:
        gold_vid = gold_vid if gold_vid is not None else variant_ids[0]
        steps = steps if steps is not None else [0]
        palette = {vid: self._VARIANT_COLORS[idx % len(self._VARIANT_COLORS)] for idx, vid in enumerate(variant_ids)}
        fw_nodes: dict[int, AlignmentNode] = {nid: node for nid, node in agraph.nodes.items() if node.stage == Stage.FORWARD}
        bw_nodes: dict[int, AlignmentNode] = {nid: node for nid, node in agraph.nodes.items() if node.stage == Stage.BACKWARD}

        fw_edges: list[dict[str, Any]] = []
        bw_edges: list[dict[str, Any]] = []
        for edge in agraph.iter_edges():
            visible_variants = sorted(vid for vid in variant_ids if vid in edge.variant_ids)
            if not visible_variants:
                continue

            edge_payload = self._json_safe(
                {
                    "edge_id": edge.id,
                    "src": edge.src,
                    "dst": edge.dst,
                    "ops": [op.target for op in edge.ops],
                    "label": self._dot_edge_label(edge.ops),
                    "variant_ids": visible_variants,
                }
            )
            src_node = agraph.nodes.get(edge.src)
            if src_node is None:
                continue
            if src_node.stage == Stage.FORWARD:
                fw_edges.append(edge_payload)
            else:
                bw_edges.append(edge_payload)

        payload = {
            "graph_id": agraph.id,
            "variant_ids": list(variant_ids),
            "gold_vid": gold_vid,
            "steps": list(steps),
            "variant_palette": palette,
            "grad_links": [{"src": fw_id, "dst": bw_id} for fw_id, bw_id in sorted(agraph.grad_links.items())],
            "stages": [
                self._build_stage_payload(Stage.FORWARD, fw_nodes, fw_edges, gold_vid, variant_ids, steps),
                self._build_stage_payload(Stage.BACKWARD, bw_nodes, bw_edges, gold_vid, variant_ids, steps),
            ],
            "ui_state": {
                "expandedModules": {},
                "focusedModuleId": {},
                "selectedNodeId": None,
                "stageFilters": {},
            },
        }
        return self._json_safe(payload)

    def _copy_viewer_assets(self, out_dir: Path) -> None:
        asset_root = resources.files(__package__).joinpath("viewer")
        for asset in asset_root.iterdir():
            target = out_dir / asset.name
            if asset.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                self._copy_viewer_tree(asset, target)
            else:
                target.write_bytes(asset.read_bytes())

    def _copy_viewer_tree(self, source, target_dir: Path) -> None:
        for child in source.iterdir():
            target = target_dir / child.name
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                self._copy_viewer_tree(child, target)
            else:
                target.write_bytes(child.read_bytes())

    def export_viewer(
        self,
        agraph: AlignmentGraph,
        variant_ids: list[str],
        gold_vid: Optional[str] = None,
        steps: list[int] | None = None,
        out_dir: str | Path = "align_viewer",
    ) -> Path:
        payload = self.build_viewer_payload(agraph, variant_ids=variant_ids, gold_vid=gold_vid, steps=steps)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self._copy_viewer_assets(out_path)

        graph_json = out_path / "graph.json"
        graph_json.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        graph_data_js = out_path / "graph-data.js"
        graph_data_js.write_text(
            "window.__ALIGNER_GRAPH__ = "
            + _json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + ";\n",
            encoding="utf-8",
        )
        return out_path

    def _add_module_hierarchy_subgraph(
        self,
        parent: graphviz.Digraph,
        cluster_name: str,
        cluster_label: str,
        node_ids: list[int],
        children: dict[str, dict],
        stage_nodes: dict[int, AlignmentNode],
        gold_vid: str,
        variant_ids: list[str],
        steps: list[int],
    ) -> str:
        anchor_name = f"{cluster_name}__anchor"
        with parent.subgraph(name=cluster_name) as subgraph:
            subgraph.attr(
                label=cluster_label,
                style="rounded",
                color="#BFBFBF",
                fontcolor="#666666",
                fontsize="12",
                rankdir="TB",
            )
            subgraph.node(anchor_name, label="", shape="point", width="0", height="0", style="invis")
            for node_id in sorted(node_ids):
                subgraph.node(f"n{node_id}", **self._node_attrs(stage_nodes[node_id], node_id, gold_vid, variant_ids, steps))
            child_anchors: list[str] = []
            for child_name, child_tree in sorted(children.items()):
                child_cluster_name = f"cluster_{self._sanitize_cluster_id(child_tree['path'])}"
                child_anchors.append(
                    self._add_module_hierarchy_subgraph(
                        subgraph,
                        child_cluster_name,
                        child_name,
                        child_tree["nodes"],
                        child_tree["children"],
                        stage_nodes,
                        gold_vid,
                        variant_ids,
                        steps,
                    )
                )
            for prev_anchor, next_anchor in zip(child_anchors, child_anchors[1:]):
                subgraph.edge(
                    prev_anchor,
                    next_anchor,
                    style="invis",
                    weight="100",
                    minlen="2",
                )
        return anchor_name

    def _add_stage_with_module_hierarchy(
        self,
        graph: graphviz.Digraph,
        stage_cluster_name: str,
        stage_label: str,
        stage_color: str,
        stage_nodes: dict[int, AlignmentNode],
        gold_vid: str,
        variant_ids: list[str],
        steps: list[int],
    ) -> None:
        module_tree: dict[str, dict] = {}
        root_node_ids: list[int] = []

        for node_id, align_node in sorted(stage_nodes.items()):
            path = self._module_stack_to_path(align_node.module_stack)
            if not path:
                root_node_ids.append(node_id)
                continue

            cursor = module_tree
            prefix: list[str] = []
            leaf = None
            for part in path:
                prefix.append(part)
                leaf = cursor.setdefault(
                    part,
                    {"path": tuple(prefix), "nodes": [], "children": {}},
                )
                cursor = leaf["children"]

            if leaf is not None:
                leaf["nodes"].append(node_id)

        with graph.subgraph(name=stage_cluster_name) as stage_graph:
            stage_graph.attr(
                label=stage_label,
                style="solid",
                color=stage_color,
                fontcolor=stage_color,
                fontsize="18",
                rankdir="TB",
            )
            for node_id in sorted(root_node_ids):
                stage_graph.node(f"n{node_id}", **self._node_attrs(stage_nodes[node_id], node_id, gold_vid, variant_ids, steps))
            child_anchors: list[str] = []
            for child_name, child_tree in sorted(module_tree.items()):
                child_anchors.append(
                    self._add_module_hierarchy_subgraph(
                        stage_graph,
                        f"cluster_{self._sanitize_cluster_id(child_tree['path'])}",
                        child_name,
                        child_tree["nodes"],
                        child_tree["children"],
                        stage_nodes,
                        gold_vid,
                        variant_ids,
                        steps,
                    )
                )
            for prev_anchor, next_anchor in zip(child_anchors, child_anchors[1:]):
                stage_graph.edge(
                    prev_anchor,
                    next_anchor,
                    style="invis",
                    weight="100",
                    minlen="2",
                )

    def export_dot(self, agraph: AlignmentGraph, variant_ids: list[str], gold_vid: Optional[str] = None, steps: list[int] | None = None, fpath: str = "align_graph.dot") -> graphviz.Digraph:
        gold_vid = gold_vid if gold_vid is not None else variant_ids[0]
        steps = steps if steps is not None else [0]
        fw_nodes: dict[int, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.FORWARD}
        bw_nodes: dict[int, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.BACKWARD}
        palette = {vid: self._VARIANT_COLORS[idx % len(self._VARIANT_COLORS)] for idx, vid in enumerate(variant_ids)}

        g = graphviz.Digraph(
            "AlignmentGraph",
            graph_attr={"rankdir": "TB", "newrank": "true", "fontname": "Helvetica", "fontsize": "12"},
            node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Courier", "fontsize": "10"},
            edge_attr={"fontsize": "9"},
        )

        with g.subgraph(name="cluster_forward") as fw:
            fw.attr(label="Forward", style="solid", color="#4472C4", fontcolor="#4472C4", fontsize="18", rankdir="TB")
            fw.node("forward__anchor", label="", shape="point", width="0", height="0", style="invis")
            for nid in sorted(fw_nodes):
                fw.node(f"n{nid}", **self._node_attrs(fw_nodes[nid], nid, gold_vid, variant_ids, steps))

        with g.subgraph(name="cluster_backward") as bw:
            bw.attr(label="Backward", style="solid", color="#ED7D31", fontcolor="#ED7D31", fontsize="18", rankdir="TB")
            bw.node("backward__anchor", label="", shape="point", width="0", height="0", style="invis")
            for nid in sorted(bw_nodes):
                bw.node(f"n{nid}", **self._node_attrs(bw_nodes[nid], nid, gold_vid, variant_ids, steps))

        with g.subgraph() as subgraph:
            # Keep the forward/backward stage clusters side-by-side while the graph itself flows top-to-bottom.
            subgraph.attr(rank="same")
            subgraph.node("forward__anchor")
            subgraph.node("backward__anchor")

        g.edge("forward__anchor", "backward__anchor", style="invis", weight="100", constraint="false")

        for fw_id, bw_id in sorted(agraph.grad_links.items()):
            with g.subgraph() as subgraph:
                # Encourage linked forward/backward nodes to stay on the same horizontal row.
                subgraph.attr(rank="same")
                subgraph.node(f"n{fw_id}")
                subgraph.node(f"n{bw_id}")

        for edge in agraph.iter_edges():
            visible_variants = sorted(vid for vid in variant_ids if vid in edge.variant_ids)
            if not visible_variants:
                continue
            ops_label = "\\n".join(op.target for op in edge.ops) if edge.ops else "<empty>"
            tooltip = f"variants={', '.join(visible_variants)}\nops:\n{ops_label}"
            present = [vid for vid in variant_ids if vid in edge.variant_ids]
            color = "#9E9E9E" if len(present) != 1 else palette[present[0]]
            edge_src = f"n{edge.src}"
            edge_dst = f"n{edge.dst}"
            edge_attrs = {
                "color": color,
                "fontcolor": color,
                "label": self._dot_edge_label(edge.ops),
                "tooltip": tooltip,
                "comment": _json.dumps(
                    {
                        "src_id": edge.src,
                        "dst_id": edge.dst,
                        "variant_ids": visible_variants,
                        "ops": [op.target for op in edge.ops],
                    },
                    ensure_ascii=False,
                ),
            }

            if agraph.nodes[edge.src].stage == Stage.BACKWARD and agraph.nodes[edge.dst].stage == Stage.BACKWARD:
                edge_src, edge_dst = edge_dst, edge_src
                edge_attrs["dir"] = "back"

            g.edge(
                edge_src,
                edge_dst,
                **edge_attrs,
            )

        for fw_id, bw_id in sorted(agraph.grad_links.items()):
            g.edge(f"n{fw_id}", f"n{bw_id}", style="dotted", color="#000000", constraint="false")

        g.save(fpath)
        print(f"DOT file written to {fpath}")
        return g

    def export_dot_with_module_hierarchy(
        self,
        agraph: AlignmentGraph,
        variant_ids: list[str],
        gold_vid: Optional[str] = None,
        steps: list[int] | None = None,
        fpath: str = "align_graph_module_hierarchy.dot",
    ) -> graphviz.Digraph:
        gold_vid = gold_vid if gold_vid is not None else variant_ids[0]
        steps = steps if steps is not None else [0]
        fw_nodes: dict[int, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.FORWARD}
        bw_nodes: dict[int, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.BACKWARD}
        palette = {vid: self._VARIANT_COLORS[idx % len(self._VARIANT_COLORS)] for idx, vid in enumerate(variant_ids)}

        g = graphviz.Digraph(
            "AlignmentGraph",
            graph_attr={"rankdir": "TB", "newrank": "true", "fontname": "Helvetica", "fontsize": "12"},
            node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Courier", "fontsize": "10"},
            edge_attr={"fontsize": "9"},
        )

        self._add_stage_with_module_hierarchy(
            g,
            "cluster_forward",
            "Forward",
            "#4472C4",
            fw_nodes,
            gold_vid,
            variant_ids,
            steps,
        )
        self._add_stage_with_module_hierarchy(
            g,
            "cluster_backward",
            "Backward",
            "#ED7D31",
            bw_nodes,
            gold_vid,
            variant_ids,
            steps,
        )

        for fw_id, bw_id in sorted(agraph.grad_links.items()):
            with g.subgraph() as subgraph:
                subgraph.attr(rank="same")
                subgraph.node(f"n{fw_id}")
                subgraph.node(f"n{bw_id}")

        for edge in agraph.iter_edges():
            visible_variants = sorted(vid for vid in variant_ids if vid in edge.variant_ids)
            if not visible_variants:
                continue
            ops_label = "\\n".join(op.target for op in edge.ops) if edge.ops else "<empty>"
            tooltip = f"variants={', '.join(visible_variants)}\nops:\n{ops_label}"
            present = [vid for vid in variant_ids if vid in edge.variant_ids]
            color = "#9E9E9E" if len(present) != 1 else palette[present[0]]
            g.edge(
                f"n{edge.src}",
                f"n{edge.dst}",
                color=color,
                fontcolor=color,
                label=self._dot_edge_label(edge.ops),
                tooltip=tooltip,
                comment=_json.dumps(
                    {
                        "src_id": edge.src,
                        "dst_id": edge.dst,
                        "variant_ids": visible_variants,
                        "ops": [op.target for op in edge.ops],
                    },
                    ensure_ascii=False,
                ),
            )

        for fw_id, bw_id in sorted(agraph.grad_links.items()):
            g.edge(f"n{fw_id}", f"n{bw_id}", style="dotted", color="#000000", constraint="false")

        g.save(fpath)
        print(f"DOT file written to {fpath}")
        return g
