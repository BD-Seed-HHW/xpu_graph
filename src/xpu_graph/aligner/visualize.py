import json as _json
from typing import Optional

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

    def export_dot(self, agraph: AlignmentGraph, variant_ids: list[str], gold_vid: Optional[str] = None, steps: list[int] | None = None, fpath: str = "align_graph.dot") -> graphviz.Digraph:
        gold_vid = gold_vid if gold_vid is not None else variant_ids[0]
        steps = steps if steps is not None else [0]
        fw_nodes: dict[int, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.FORWARD}
        bw_nodes: dict[int, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.BACKWARD}
        palette = {vid: self._VARIANT_COLORS[idx % len(self._VARIANT_COLORS)] for idx, vid in enumerate(variant_ids)}

        def _build_node_attrs(node_id: int, align_node: AlignmentNode) -> dict[str, str]:
            op_meta = align_node.get_meta("op_meta", gold_vid, {})
            opinfo = op_meta.get("last_non_trivial_op") or op_meta.get("last_op")
            op_name = opinfo.target if isinstance(opinfo, OpInfo) else "?"
            label = f"{node_id}: {op_name}"

            gold_data = align_node.data.get(gold_vid, [])
            all_step_details: list[dict] = []
            for s in steps:
                step_variants: list[dict] = []
                for vid in variant_ids:
                    entry: dict = {"variant": vid}
                    data = align_node.data.get(vid, [])
                    if not data or s >= len(data):
                        entry["status"] = "no data"
                    else:
                        xorsum, raw = data[s][0], data[s][1]
                        raw_32 = raw.to(torch.float32)
                        entry["dtype"] = str(raw.dtype).replace("torch.", "")
                        entry["xorsum"] = f"0x{xorsum:08X}"
                        if gold_data and s < len(gold_data):
                            g32 = gold_data[s][1].to(torch.float32)
                            r32 = raw_32.to(torch.float32)
                            entry["max_diff"] = float(f"{(r32 - g32).abs().max().item():.8f}")
                            entry["closeto"] = bool(torch.allclose(r32, g32, rtol=1e-3, atol=1e-5))
                    step_variants.append(entry)
                all_step_details.append({"step": s, "variants": step_variants})

            nbsp = "\u00A0"
            tip_lines = [f"{'-'*80}\nNode {node_id} - {op_name}\n{'-'*80}"]
            for group in all_step_details:
                tip_lines.append(f"Step {group['step']}:")
                for d in group["variants"]:
                    if d.get("status") == "no data":
                        tip_lines.append(f"  {d['variant']}: no data")
                    else:
                        flag = "✓" if d.get("closeto", False) else "✗"
                        gold_sign = " (gold)" if d["variant"] == gold_vid else ""
                        tip_lines.append(
                            f"  {d['variant']+':':15} {d.get('dtype','?'):8} "
                            f"xor={d.get('xorsum','?'):11} "
                            f"Δ={d.get('max_diff', '?'):.9f} {flag}{gold_sign}"
                        )
            tooltip = "\n".join(tip_lines).replace(" ", nbsp)
            comment = _json.dumps(
                {"node_id": node_id, "stage": align_node.stage.name, "op_name": op_name, "steps": all_step_details},
                ensure_ascii=False,
            )

            color = "#4472C4" if align_node.stage == Stage.FORWARD else "#AD683A"
            return {
                "label": label,
                "tooltip": tooltip,
                "comment": comment,
                "style": "filled",
                "fillcolor": "#DEEBF7" if align_node.stage == Stage.FORWARD else "#FBE5D6",
                "color": color,
                "fontcolor": "#333333",
            }

        g = graphviz.Digraph(
            "AlignmentGraph",
            graph_attr={"rankdir": "TB", "newrank": "true", "fontname": "Helvetica", "fontsize": "12"},
            node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Courier", "fontsize": "10"},
            edge_attr={"fontsize": "9"},
        )

        with g.subgraph(name="cluster_forward") as fw:
            fw.attr(label="Forward", style="solid", color="#4472C4", fontcolor="#4472C4", fontsize="18")
            for nid in sorted(fw_nodes):
                fw.node(f"n{nid}", **_build_node_attrs(nid, fw_nodes[nid]))

        with g.subgraph(name="cluster_backward") as bw:
            bw.attr(label="Backward", style="solid", color="#ED7D31", fontcolor="#ED7D31", fontsize="18")
            for nid in sorted(bw_nodes):
                bw.node(f"n{nid}", **_build_node_attrs(nid, bw_nodes[nid]))

        for fw_id, bw_id in sorted(agraph.grad_links.items()):
            with g.subgraph() as s:
                s.attr(rank="same")
                s.node(f"n{fw_id}")
                s.node(f"n{bw_id}")

        for edge in agraph.iter_edges():
            visible_variants = sorted(vid for vid in variant_ids if vid in edge.variant_ids)
            if not visible_variants:
                continue
            ops_label = "\\n".join(op.target for op in edge.ops) if edge.ops else "<empty>"
            tooltip = f"variants={', '.join(visible_variants)}\nops:\n{ops_label}"
            present = [vid for vid in variant_ids if vid in edge.variant_ids]
            color = "#9E9E9E" if len(present) != 1 else palette[present[0]]
            g.edge(
                f"n{edge.src_id}",
                f"n{edge.dst_id}",
                color=color,
                fontcolor=color,
                label="|".join(
                    (
                        op.target.split("aten::", 1)[1].split(".", 1)[0]
                        if "aten::" in op.target
                        else op.target.split("::", 1)[1].split(".", 1)[0]
                        if "::" in op.target
                        else op.target
                    )
                    for op in edge.ops
                ),
                tooltip=tooltip,
                comment=_json.dumps(
                    {
                        "src_id": edge.src_id,
                        "dst_id": edge.dst_id,
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
