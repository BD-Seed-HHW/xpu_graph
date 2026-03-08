from typing import Optional
import numpy as np
import json as _json
import graphviz

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch._functorch.aot_autograd import aot_module_simplified
import torch._dynamo as dynamo
import functorch

from .alignment_graph import AlignmentNode, AlignmentGraph, GraphMapping, Stage


class AlignmentManager:
    _instance: Optional["AlignmentManager"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if AlignmentManager._initialized:
            return
        self._graphs: dict[str, AlignmentGraph] = {}
        self._mappings: list[GraphMapping] = []
        self._register_ops()
        AlignmentManager._initialized = True

    @property
    def graphs(self) -> dict[str, AlignmentGraph]:
        return dict(self._graphs)
    @property
    def mappings(self) -> list[GraphMapping]:
        return list(self._mappings)

    def get_graph(self, id: str) -> AlignmentGraph:
        if id not in self._graphs:
            self._graphs[id] = AlignmentGraph(id)
        return self._graphs[id]

    def _register_ops(self):
        mgr = self
        # -- ops.xpugraph.marker --
        @torch.library.custom_op("xpugraph::marker", mutates_args=())
        def xpugraph_marker(x: torch.Tensor, nid: int) -> torch.Tensor:
            return x.clone()
        @xpugraph_marker.register_fake
        def xpugraph_marker_fake(x: torch.Tensor, nid: int) -> torch.Tensor:
            return x.clone()
        def xpugraph_marker_setup_context(ctx, inputs, output):
            ctx.fw_nid = inputs[1]
        def xpugraph_marker_backward(ctx, grad_output):
            return torch.ops.xpugraph.marker(grad_output, ctx.fw_nid), None
        xpugraph_marker.register_autograd(
            xpugraph_marker_backward,
            setup_context=xpugraph_marker_setup_context,
        )

        # -- ops.xpugraph.instrument --
        @torch.library.custom_op("xpugraph::instrument", mutates_args=())
        def xpugraph_instrument(x: torch.Tensor, nid: int, stage: int, gid: str, vid: str) -> torch.Tensor:
            mgr.get_graph(gid).get_node(nid, stage).record(
                vid,
                self.xorsum32(x),
                x.detach().cpu()
            )
            return torch.empty(0)
        @xpugraph_instrument.register_fake
        def xpugraph_instrument_fake(x: torch.Tensor, nid: int, stage: int, gid: str, vid: str) -> torch.Tensor:
            return torch.empty(0)
        xpugraph_instrument.register_autograd(
            lambda ctx, grad_output: (None, None, None, None, None),
            setup_context=lambda ctx, inputs, output: None
        )

    @staticmethod
    def xorsum32(t: torch.Tensor) -> int:
        """32-bit XOR checksum of the tensor's raw bytes, padded to a multiple of 4 bytes if necessary."""
        b = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        b = F.pad(b, (0, (-b.numel()) % 4))
        w = b.view(-1, 4).to(torch.int64)
        u32 = w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)
        return int(np.bitwise_xor.reduce(u32.numpy(), dtype=np.uint64)) & 0xFFFFFFFF

    def inject_marker_meta_and_remove_marker_fw_pass(self, agraph_id: str, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fxgraph: torch.fx.Graph = gm.graph
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        markers_to_remove = []

        for fxnode in fxgraph.nodes:
            if fxnode.op == "call_function" and fxnode.target == torch.ops.xpugraph.marker.default:
                source_node: torch.fx.Node = fxnode.args[0]
                nid: int = fxnode.args[1]
                agraph.get_node(nid, Stage.FORWARD).meta.update({
                    "op": source_node.op,
                    "name": source_node.name,
                    "target": source_node.target,
                })
                source_node.meta.setdefault("align_node_ids", []).append(nid)
                fxnode.replace_all_uses_with(source_node)
                markers_to_remove.append(fxnode)

        for fxnode in markers_to_remove:
            fxgraph.erase_node(fxnode)

        fxgraph.lint()
        gm.recompile()
        return gm

    def inject_marker_meta_and_remove_marker_bw_pass(self, agraph_id: str, bw_nid_offs: int, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fxgraph: torch.fx.Graph = gm.graph
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        markers_to_remove = []

        for fxnode in fxgraph.nodes:
            if fxnode.op == "call_function" and fxnode.target == torch.ops.xpugraph.marker.default:
                source_node: torch.fx.Node = fxnode.args[0]
                fw_nid: int = fxnode.args[1]
                nid = fw_nid + bw_nid_offs  # bw align nid typically is offset from fw nid by total number of fw markers.
                agraph.add_grad_link(fw_nid, nid)
                agraph.get_node(nid, Stage.BACKWARD).meta.update({
                    "op": source_node.op,
                    "name": source_node.name,
                    "target": source_node.target,
                })
                source_node.meta.setdefault("align_node_ids", []).append(nid)
                fxnode.replace_all_uses_with(source_node)
                markers_to_remove.append(fxnode)

        for fxnode in markers_to_remove:
            fxgraph.erase_node(fxnode)

        fxgraph.lint()
        gm.recompile()
        return gm

    def insert_instrument_nodes_pass(self, stage: Stage, agraph_id: str, variant_id: str, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Insert ``ops.xpugraph.instrument`` calls after every anchored ``fx.Node``."""
        fxgraph: torch.fx.Graph = gm.graph

        for node in list(fxgraph.nodes):
            for node_id in node.meta.get("align_node_ids", []):
                with fxgraph.inserting_after(node):
                    fxgraph.call_function(
                        torch.ops.xpugraph.instrument.default,
                        args=(node, node_id, stage, agraph_id, variant_id),
                    )

        fxgraph.lint()
        gm.recompile()
        return gm

    def build_topology_from_fxgraph(self, agraph_id: str, fxgraph: torch.fx.Graph):
        """Build topology edges via dataflow coloring on a topologically-ordered ``fx.Graph``."""
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        reaching: dict[torch.fx.Node, set[int]] = {}

        for node in fxgraph.nodes:
            incoming: set[int] = set()
            for inp in node.all_input_nodes:
                incoming |= reaching.get(inp, set())

            own_ids = set(node.meta.get("align_node_ids", []))
            if own_ids:
                for s in incoming:
                    for d in own_ids:
                        if s != d:
                            agraph.add_edge(s, d)
                reaching[node] = own_ids
            else:
                reaching[node] = incoming

    def print_data(self, agraph_id: str, variant_ids: list[str], gold_vid: Optional[str] = None) -> None:
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        gold_vid: str = gold_vid if gold_vid is not None else variant_ids[0]
        nodes: list[AlignmentNode] = agraph.nodes

        n_steps = max(len(n.data.get(gold_vid, [])) for n in nodes.values())
        for i in range(n_steps):
            print(f"\n\n{'='*100}\nStep {i}  —  Graph {agraph.id!r}\n{'='*100}")
            for node_id, align_node in nodes.items():
                nmeta, ndata = align_node.meta, align_node.data
                print(f"\nNode {node_id} - {align_node.stage.name} {nmeta['op']} {nmeta['target']}")
                gold_xor, gold_ten = ndata[gold_vid][i][0], ndata[gold_vid][i][1].to(torch.float32)
                for vid in variant_ids:
                    data = ndata.get(vid, [])
                    if not data or i >= len(data):
                        print(f"  {vid}: No data")
                        continue
                    var_xor, var_ten = data[i][0], data[i][1]
                    max_diff = float((var_ten.float() - gold_ten).abs().max().item())
                    closeto = bool(torch.allclose(var_ten.float(), gold_ten, atol=1e-8, rtol=1e-4))
                    gold_tag = " (gold)" if vid == gold_vid else ""
                    print(
                        f"  {vid + gold_tag:20s}: dtype={var_ten.dtype}, "
                        f"xorsum=0x{var_xor:08X}, max_diff={max_diff:.10f}, "
                        f"closeto={str(closeto):5}"
                    )

    def export_dot(self, agraph_id: str, variant_ids: list[str], gold_vid: Optional[str] = None, steps: list[int] = None, fpath: str = "align_graph.dot") -> graphviz.Digraph:
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        gold_vid: str = gold_vid if gold_vid is not None else variant_ids[0]
        steps: list[int] = steps if steps is not None else [0]
        fw_nodes: dict[str, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.FORWARD}
        bw_nodes: dict[str, AlignmentNode] = {nid: n for nid, n in agraph.nodes.items() if n.stage == Stage.BACKWARD}

        def _build_node_attrs(node_id: int, align_node: AlignmentNode) -> dict[str, str]:
            op_name = str(align_node.meta.get("target", "?"))
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
                        xorsum, raw_32 = data[s][0], data[s][1].to(torch.float32)
                        entry["dtype"] = str(raw_32.dtype).replace("torch.", "")
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
                        flag = '✓' if d.get('closeto', False) else '✗'
                        gold_sign = ' (gold)' if d['variant'] == gold_vid else ''
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

        g = graphviz.Digraph("AlignmentGraph",
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

        for src_id, dst_ids in sorted(agraph.topology.items()):
            for dst_id in sorted(set(dst_ids)):
                g.edge(f"n{src_id}", f"n{dst_id}")

        for fw_id, bw_id in sorted(agraph.grad_links.items()):
            g.edge(f"n{fw_id}", f"n{bw_id}", style="dotted", color="#000000", constraint="false")

        g.save(fpath)
        print(f"DOT file written to {fpath}")
        return g


class AlignedModelGenerator:
    """Produces one instrumented model variant (compiled **or** eager). One instance could only use once."""
    class MarkerInjectFunctionMode(TorchFunctionMode):
        def __init__(self, ctx: 'AlignedModelGenerator'):
            super().__init__()
            self._ctx = ctx
            self._next_node_id: int = 0
            self._disabled: bool = False

        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if not self._disabled:
                result = self._inject_markers(result)
            return result

        def _inject_markers(self, result):
            if isinstance(result, torch.Tensor):
                if result.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    return torch.ops.xpugraph.marker(result, self._new_node_id())
            elif isinstance(result, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in result):
                items = [self._inject_markers(item) for item in result]
                return type(result)(items)
            return result
        
        def _new_node_id(self) -> int:
            ret = self._next_node_id
            self._next_node_id += 1
            return ret

    class EagerInstrumentFunctionMode(TorchFunctionMode):
        def __init__(self, ctx: 'AlignedModelGenerator'):
            super().__init__()
            self._ctx = ctx
            self._next_node_id: int = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            self._instrument(result)
            return result

        def _instrument(self, result):
            if isinstance(result, torch.Tensor):
                if result.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    node_id = self._next_node_id
                    self._next_node_id += 1
                    torch.ops.xpugraph.instrument(
                        result,
                        node_id,
                        Stage.FORWARD,
                        self._ctx._agraph_id,
                        self._ctx._variant_id
                    )
                    if result.requires_grad:
                        def _bw_hook(grad, nid=node_id, mode=self):
                            bw_nid = nid + mode._next_node_id  # bw nid typically offset from fw nid by total number of fw markers (e.g., _next_node_id).
                            mode._ctx._agraph.add_grad_link(nid, bw_nid)
                            torch.ops.xpugraph.instrument(
                                grad,
                                bw_nid,
                                Stage.BACKWARD,
                                mode._ctx._agraph_id,
                                mode._ctx._variant_id
                            )
                        result.register_hook(_bw_hook)
            elif isinstance(result, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in result):
                for item in result:
                    self._instrument(item)
        
        def reset_id_counter(self):
            self._next_node_id = 0

    def __init__(self, agraph_id: str, variant_id: str):
        self._mgr: AlignmentManager = AlignmentManager()
        self._agraph_id:str = agraph_id
        self._variant_id:str = variant_id
        self._agraph: AlignmentGraph = self._mgr.get_graph(agraph_id)
        self._marker_inject_mode = self.MarkerInjectFunctionMode(self)
        self._eager_instrument_mode = self.EagerInstrumentFunctionMode(self)
        self._generated: bool = False
        self._agraph.register_variant(variant_id)

    def _make_backend(self):
        fw_marker_count = [0]   # shared between fw/bw compilers
        variant_id = self._variant_id

        def alignment_fw_compiler(gm: torch.fx.GraphModule, example_inputs):
            fw_marker_count[0] = sum(1 for n in gm.graph.nodes if n.target == torch.ops.xpugraph.marker.default)
            gm = self._mgr.inject_marker_meta_and_remove_marker_fw_pass(self._agraph_id, gm)
            self._mgr.build_topology_from_fxgraph(self._agraph_id, gm.graph)
            # any optimization pass...
            gm = self._mgr.insert_instrument_nodes_pass(Stage.FORWARD, self._agraph_id, variant_id, gm)
            return functorch.compile.make_boxed_func(gm.forward)

        def alignment_bw_compiler(gm: torch.fx.GraphModule, example_inputs):
            bw_offset = fw_marker_count[0]
            gm = self._mgr.inject_marker_meta_and_remove_marker_bw_pass(self._agraph_id, bw_offset, gm)
            self._mgr.build_topology_from_fxgraph(self._agraph_id, gm.graph)
            # any optimization pass...
            gm = self._mgr.insert_instrument_nodes_pass(Stage.BACKWARD, self._agraph_id, variant_id, gm)
            return functorch.compile.make_boxed_func(gm.forward)

        def backend(gm: torch.fx.GraphModule, example_inputs):
            gm = aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=alignment_fw_compiler,
                bw_compiler=alignment_bw_compiler,
            )
            # disable dynamo guards to avoid recompile.
            tracing_context = torch._guards.TracingContext.try_get()
            tracing_context.guards_context.dynamo_guards = (torch._guards.GuardsSet(set()))  # TODO: filter relevant guards only.
            return gm

        return backend

    def get_compiled(self, model: nn.Module, example_inputs: tuple) -> nn.Module:
        """Return a compiled, instrumented model."""
        if self._generated:
            raise RuntimeError("An AlignedModelGenerator instance can only generate once. Create a new one for another generation.")

        with self._marker_inject_mode:
            compiled_fn = torch.compile(model, backend=self._make_backend(), dynamic=True, fullgraph=True)
            _ = compiled_fn(*example_inputs)
            _.backward(torch.ones_like(_))
        self._marker_inject_mode._disabled = True
        # clear recorded data during warmup
        self._mgr.get_graph(self._agraph_id).clear_data()

        wrapper_marker_mode = self._marker_inject_mode  # prevent closure over self
        class _WrapModuleCompile(nn.Module):
            def __init__(wself, fn):
                super().__init__()
                wself.fn = fn

            @dynamo.disable
            def forward(wself, *args, **kwargs):
                with wrapper_marker_mode:
                    return wself.fn(*args, **kwargs)
        wrapped_model = _WrapModuleCompile(compiled_fn)

        self._generated = True
        return wrapped_model

    def get_eager(self, model: nn.Module) -> nn.Module:
        """Return an eager-mode instrumented model."""
        if self._generated:
            raise RuntimeError("An AlignedModelGenerator instance can only generate once. Create a new one for another generation.")

        eager_inst_mode = self._eager_instrument_mode
        class _WrapModule(nn.Module):
            def __init__(wself, mdl):
                super().__init__()
                wself._model = mdl

            def forward(wself, *args, **kwargs):
                eager_inst_mode.reset_id_counter()  # reset node id counter for each forward to keep them consistent across variants
                with eager_inst_mode:
                    return wself._model(*args, **kwargs)
        wrapped_model = _WrapModule(model)

        self._generated = True
        return wrapped_model