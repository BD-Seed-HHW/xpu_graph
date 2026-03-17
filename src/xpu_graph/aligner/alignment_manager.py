from typing import Optional

import functorch
import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch._functorch.aot_autograd import aot_module_simplified
from torch.overrides import TorchFunctionMode
import torch.utils._pytree as pytree

from .alignment_graph import AlignmentGraph, GraphMapping, OpInfo, Stage
from .visualize import AlignmentVisualizer


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

    def get_graph(self, gid: str) -> AlignmentGraph:
        if gid not in self._graphs:
            self._graphs[gid] = AlignmentGraph(gid)
        return self._graphs[gid]

    def _register_ops(self):
        mgr = self
        # -- ops.xpugraph.marker --
        @torch.library.custom_op("xpugraph::marker", mutates_args=())
        def xpugraph_marker(x: torch.Tensor, nid: int) -> torch.Tensor:
            return x.clone()
        @xpugraph_marker.register_fake
        def xpugraph_marker_fake(x: torch.Tensor, nid: int) -> torch.Tensor:
            return torch.empty_like(x)
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
            mgr.get_graph(gid).get_node(nid, Stage(stage)).record_data(
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

    # @staticmethod
    # def xorsum32(t: torch.Tensor) -> int:
    #     """32-bit XOR checksum of the tensor's raw bytes, padded to a multiple of 4 bytes if necessary."""
    #     b = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    #     b = F.pad(b, (0, (-b.numel()) % 4))
    #     w = b.view(-1, 4).to(torch.int64)
    #     u32 = w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)
    #     return int(np.bitwise_xor.reduce(u32.numpy(), dtype=np.uint64)) & 0xFFFFFFFF

    @staticmethod
    def xorsum32(t: torch.Tensor) -> int:
        t = t.detach()
        b = t.contiguous().view(torch.uint8).reshape(-1)
        pad_bytes = (-b.numel()) % 4
        if pad_bytes > 0:
            b = F.pad(b, (0, pad_bytes))
        # View as 32-bit integers
        # Note: view(int32) uses machine endianness (usually LE), which matches the original manual LE construction on standard platforms.
        u32 = b.view(torch.int32)
        while u32.numel() > 1:
            if u32.numel() % 2 != 0:
                u32 = F.pad(u32, (0, 1))
            u32 = u32.view(-1, 2)
            u32 = torch.bitwise_xor(u32[:, 0], u32[:, 1])
        if u32.numel() == 0:
            return 0
        return u32.item() & 0xFFFFFFFF

    def _find_last_non_trivial_op(self, fxnode: torch.fx.Node) -> OpInfo | None:
        trivial_target_suffixes = (
            "view",
            "reshape",
            "_unsafe_view",
            "to",
            "type_as",
            "contiguous",
            "clone",
            "_to_copy",
            "cpu",
            "float",
            "double",
            "half",
            "bfloat16",
        )
        source_op = OpInfo.from_fx_node(fxnode)
        if not any(source_op.target.endswith(suffix) for suffix in trivial_target_suffixes):
            return source_op

        for input_node in fxnode.all_input_nodes:
            candidate = self._find_last_non_trivial_op(input_node)
            if candidate is not None:
                return candidate
        return None

    def inject_marker_meta_and_remove_marker_fw_pass(self, agraph_id: str, variant_id: str, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fxgraph: torch.fx.Graph = gm.graph
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        markers_to_remove = []

        for fxnode in fxgraph.nodes:
            if fxnode.op == "call_function" and fxnode.target == torch.ops.xpugraph.marker.default:
                source_node: torch.fx.Node = fxnode.args[0]
                nid: int = fxnode.args[1]
                align_node_ids = source_node.meta.setdefault("align_node_ids", [])
                if not align_node_ids:
                    align_node_ids.append(nid)
                    agraph.get_node(nid, Stage.FORWARD).set_meta("op_meta", variant_id,{
                            "last_op": OpInfo.from_fx_node(source_node),
                            "last_non_trivial_op": self._find_last_non_trivial_op(source_node),
                    })
                else:
                    source_node.meta.setdefault("collapsed_align_node_ids", []).append(nid)
                    agraph.mark_collapsed_fw_node(variant_id, nid)
                fxnode.replace_all_uses_with(source_node)
                markers_to_remove.append(fxnode)

        for fxnode in markers_to_remove:
            fxgraph.erase_node(fxnode)

        fxgraph.lint()
        gm.recompile()
        return gm

    def build_canonical_module_stack_pass(
        self,
        agraph_id: str,
        stage: Stage,
        gm: torch.fx.GraphModule,
        bw_nid_offs: int = 0,
    ) -> torch.fx.GraphModule:
        """Persist one canonical raw ``nn_module_stack`` per ``AlignmentNode``."""
        fxgraph: torch.fx.Graph = gm.graph
        agraph: AlignmentGraph = self.get_graph(agraph_id)

        for fxnode in fxgraph.nodes:
            if fxnode.op != "call_function" or fxnode.target != torch.ops.xpugraph.marker.default:
                continue

            source_node: torch.fx.Node = fxnode.args[0]
            fw_nid: int = fxnode.args[1]
            nid = fw_nid if stage == Stage.FORWARD else fw_nid + bw_nid_offs

            module_stack = source_node.meta.get("nn_module_stack", fxnode.meta.get("nn_module_stack"))
            if module_stack is not None:
                align_node = agraph.get_node(nid, stage)
                if align_node.module_stack is None:
                    align_node.module_stack = module_stack
                else:
                    pass
                    # assert (
                    #     align_node.module_stack == module_stack
                    # ), f"AlignmentNode {nid} has inconsistent module_stack across variants"

        return gm

    def inject_marker_meta_and_remove_marker_bw_pass(self, agraph_id: str, variant_id: str, bw_nid_offs: int, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fxgraph: torch.fx.Graph = gm.graph
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        markers_to_remove = []

        for fxnode in fxgraph.nodes:
            if fxnode.op == "call_function" and fxnode.target == torch.ops.xpugraph.marker.default:
                source_node: torch.fx.Node = fxnode.args[0]
                fw_nid: int = fxnode.args[1]
                nid = fw_nid + bw_nid_offs  # bw align nid typically is offset from fw nid by total number of fw markers.
                if agraph.is_fw_node_collapsed(variant_id, fw_nid):
                    source_node.meta.setdefault("collapsed_align_node_ids", []).append(nid)
                    fxnode.replace_all_uses_with(source_node)
                    markers_to_remove.append(fxnode)
                    continue
                source_node.meta.setdefault("align_node_ids", []).append(nid)
                agraph.add_grad_link(fw_nid, nid)
                agraph.get_node(nid, Stage.BACKWARD).set_meta("op_meta", variant_id, {
                        "last_op": OpInfo.from_fx_node(source_node),
                        "last_non_trivial_op": self._find_last_non_trivial_op(source_node),
                })
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

    def build_topology_from_fxgraph(self, agraph_id: str, variant_id: str, fxgraph: torch.fx.Graph):
        """Build variant-aware topology edges from a topologically-ordered ``fx.Graph``."""
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        reaching: dict[torch.fx.Node, dict[int, set[tuple[OpInfo, ...]]]] = {}

        def _append_current_op_to_paths(fxnode: torch.fx.Node, paths: dict[int, set[tuple[OpInfo, ...]]]) -> dict[int, set[tuple[OpInfo, ...]]]:
            if not paths:
                return {}
            opinfo = OpInfo.from_fx_node(fxnode)
            return {src_id: {path + (opinfo,) for path in path_set} for src_id, path_set in paths.items()}

        for fxnode in fxgraph.nodes:
            incoming: dict[int, set[tuple[OpInfo, ...]]] = {}
            for input_node in fxnode.all_input_nodes:
                for src_id, path_set in reaching.get(input_node, {}).items():
                    incoming.setdefault(src_id, set()).update(path_set)

            with_current = _append_current_op_to_paths(fxnode, incoming)
            own_ids = tuple(dict.fromkeys(fxnode.meta.get("align_node_ids", [])))
            if own_ids:
                for src_id, path_set in with_current.items():
                    for dst_id in own_ids:
                        if src_id == dst_id:
                            continue
                        for path in path_set:
                            agraph.add_edge(variant_id, src_id, dst_id, path)
                reaching[fxnode] = {dst_id: {()} for dst_id in own_ids}
            else:
                reaching[fxnode] = with_current

    def print_data(self, agraph_id: str, variant_ids: list[str], gold_vid: Optional[str] = None) -> None:
        agraph: AlignmentGraph = self.get_graph(agraph_id)
        gold_vid = gold_vid if gold_vid is not None else variant_ids[0]

        n_steps = 0 if len(agraph.nodes) == 0 else max(len(n.data.get(gold_vid, [])) for n in agraph.nodes.values())
        for i in range(n_steps):
            print(f"\n{'='*100}\nStep {i}  -  Graph {agraph.id!r}\n{'='*100}")
            for node_id, align_node in agraph.nodes.items():
                ndata = align_node.data
                op_desc = align_node.get_meta("op_meta", gold_vid, {}).get("last_op")
                op_text = op_desc.target if isinstance(op_desc, OpInfo) else "?"
                print(f"\nNode {node_id} - {align_node.stage.name} {op_text}")
                if i >= len(ndata.get(gold_vid, [])):
                    print(f"  {gold_vid}: No data")
                    continue
                _, gold_ten = ndata[gold_vid][i][0], ndata[gold_vid][i][1].to(torch.float32)
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
            print()

    def export_dot(self, agraph_id: str, variant_ids: list[str], gold_vid: Optional[str] = None, steps: list[int] = None, with_module_stack: bool = False, fpath: str = "align_graph.dot"):
        if with_module_stack:
            return AlignmentVisualizer().export_dot_with_module_hierarchy(self.get_graph(agraph_id), variant_ids=variant_ids, gold_vid=gold_vid, steps=steps, fpath=fpath)
        return AlignmentVisualizer().export_dot(self.get_graph(agraph_id), variant_ids=variant_ids, gold_vid=gold_vid, steps=steps, fpath=fpath)

    def export_viewer(self, agraph_id: str, variant_ids: list[str], gold_vid: Optional[str] = None, steps: list[int] | None = None, out_dir: str = "align_viewer"):
        return AlignmentVisualizer().export_viewer(
            self.get_graph(agraph_id),
            variant_ids=variant_ids,
            gold_vid=gold_vid,
            steps=steps,
            out_dir=out_dir,
        )


class AlignedModelGenerator:
    """Produces one instrumented model variant (compiled or eager). One instance could only use once."""
    class MarkerInjectFunctionMode(TorchFunctionMode):
        def __init__(self, ctx: 'AlignedModelGenerator'):
            super().__init__()
            self._ctx = ctx
            self._next_node_id: int = 0
            self._disabled: bool = False

        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if (not self._disabled and not self._ctx._is_noop(func, args, kwargs)):
                result = self._inject_markers(result)
            return result

        def _inject_markers(self, result):
            def _inject_tensor(t):
                if isinstance(t, torch.Tensor) and t.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    return torch.ops.xpugraph.marker(t, self._new_node_id())
                return t
            return pytree.tree_map(_inject_tensor, result)
        
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
            if (not self._ctx._is_noop(func, args, kwargs)):
                self._instrument(result, func)
            return result

        def reset_id_counter(self):
            self._next_node_id = 0

        def _instrument(self, result, func):
            pytree.tree_map_only(torch.Tensor, lambda tensor: self._instrument_tensor(tensor, func), result)

        def _instrument_tensor(self, result: torch.Tensor, func):
            if result.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                return
            node_id = self._next_node_id
            self._next_node_id += 1
            opinfo = OpInfo(op="eager", target=OpInfo.normalize_target(func), name=getattr(func, "__name__", None))
            self._ctx._agraph.get_node(node_id, Stage.FORWARD).set_meta("op_meta", self._ctx._variant_id, {
                    "last_op": opinfo,
                    "last_non_trivial_op": None,
            })
            torch.ops.xpugraph.instrument(result, node_id, Stage.FORWARD, self._ctx._agraph_id, self._ctx._variant_id)
            if result.requires_grad:
                result_grad_fn = result.grad_fn
                def _bw_hook(grad, nid=node_id, mode=self):
                    bw_nid = nid + mode._next_node_id  # bw nid typically offset from fw nid by total number of fw markers (e.g., _next_node_id).
                    mode._ctx._agraph.add_grad_link(nid, bw_nid)
                    target = type(result_grad_fn).__name__ if result_grad_fn is not None else "UnknownBackward"
                    bw_opinfo = OpInfo(op="autograd", target=target, name=target)
                    mode._ctx._agraph.get_node(bw_nid, Stage.BACKWARD).set_meta("op_meta", mode._ctx._variant_id, {
                            "last_op": bw_opinfo,
                            "last_non_trivial_op": None,
                    })
                    torch.ops.xpugraph.instrument(grad, bw_nid, Stage.BACKWARD, mode._ctx._agraph_id, mode._ctx._variant_id)
                result.register_hook(_bw_hook)

    @classmethod
    def _is_noop(cls, func, args, kwargs=None) -> bool:
        _NOOP_DTYPE_METHODS = {
            "float": torch.float32,
            "double": torch.float64,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if not args or not isinstance(args[0], torch.Tensor):
            return False
        x = args[0]
        func_name = getattr(func, "__name__", None)

        if func_name in _NOOP_DTYPE_METHODS:
            return x.dtype == _NOOP_DTYPE_METHODS[func_name]
        if func_name == "cpu":
            return x.device.type == "cpu"
        if func_name == "contiguous":
            kwargs = kwargs or {}
            memory_format = kwargs.get("memory_format", torch.contiguous_format)
            if memory_format is torch.contiguous_format:
                return x.is_contiguous()
            if memory_format is torch.channels_last:
                return x.is_contiguous(memory_format=torch.channels_last)
            if memory_format is torch.channels_last_3d:
                return x.is_contiguous(memory_format=torch.channels_last_3d)
            return False
        #TODO: func_name == "type_as" check will cause dynamo bug, fix and add it back.
        return False

    def __init__(self, agraph_id: str, variant_id: str):
        self._mgr: AlignmentManager = AlignmentManager()
        self._agraph_id: str = agraph_id
        self._variant_id: str = variant_id
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
            gm = self._mgr.build_canonical_module_stack_pass(self._agraph_id, Stage.FORWARD, gm)
            gm = self._mgr.inject_marker_meta_and_remove_marker_fw_pass(self._agraph_id, variant_id, gm)
            self._mgr.build_topology_from_fxgraph(self._agraph_id, variant_id, gm.graph)
            # any optimization pass...
            gm = self._mgr.insert_instrument_nodes_pass(Stage.FORWARD, self._agraph_id, variant_id, gm)
            print("="*20+"\nAfter aot_autograd (forward)\n"+"="*20)
            gm.print_readable()
            return functorch.compile.make_boxed_func(gm.forward)

        def alignment_bw_compiler(gm: torch.fx.GraphModule, example_inputs):
            bw_offset = fw_marker_count[0]
            gm = self._mgr.build_canonical_module_stack_pass(self._agraph_id, Stage.BACKWARD, gm, bw_nid_offs=bw_offset)
            gm = self._mgr.inject_marker_meta_and_remove_marker_bw_pass(self._agraph_id, variant_id, bw_offset, gm)
            self._mgr.build_topology_from_fxgraph(self._agraph_id, variant_id, gm.graph)
            # any optimization pass...
            gm = self._mgr.insert_instrument_nodes_pass(Stage.BACKWARD, self._agraph_id, variant_id, gm)
            print("="*20+"\nAfter aot_autograd (backward)\n"+"="*20)
            gm.print_readable()
            return functorch.compile.make_boxed_func(gm.forward)

        def backend(gm: torch.fx.GraphModule, example_inputs):
            print("="*20+"\nBefore aot_autograd\n"+"="*20)
            gm.print_readable(include_stride=True)
            
            gm = aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=alignment_fw_compiler,
                bw_compiler=alignment_bw_compiler,
                partition_fn=(
                    torch.functorch.partitioners.min_cut_rematerialization_partition
                    if torch.__version__ >= "2.8"
                    else torch._functorch.partitioners.default_partition
                ),
            )
            # disable dynamo guards to avoid recompile.
            tracing_context = torch._guards.TracingContext.try_get()
            tracing_context.guards_context.dynamo_guards = (torch._guards.GuardsSet(set()))  # TODO: filter relevant guards only.
            return gm

        return backend

    def get_compiled(self, model: nn.Module, args: tuple, kwargs: dict) -> nn.Module:
        """Return a compiled, instrumented model."""
        if self._generated:
            raise RuntimeError("An AlignedModelGenerator instance can only generate once. Create a new one for another generation.")

        with self._marker_inject_mode:
            compiled_fn = torch.compile(model, backend=self._make_backend(), dynamic=True, fullgraph=True)
            out = compiled_fn(*args, **kwargs)
            pytree.tree_map_only_(torch.Tensor, lambda t: t.backward(torch.ones_like(t)) if t.requires_grad else None, out)
        self._marker_inject_mode._disabled = True
        # clear recorded data during warmup
        self._mgr.get_graph(self._agraph_id).clear_data()
        # clear grad
        for p in model.parameters():
            p.grad = None

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
