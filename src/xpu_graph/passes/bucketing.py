import operator
from collections import defaultdict
from typing import Literal, Optional, TypeAlias

import torch
import torch.distributed as dist
from torch import fx

# adapted from https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/bucketing.py

BucketMode: TypeAlias = Literal["default", "custom_ops", "custom_ops_multidtype"]


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:  # type: ignore[arg-type]
    return node.op == "call_function" and (
        node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
        or node.target == torch.ops._c10d_functional.all_gather_into_tensor_out.default
    )


def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_reduce_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_reduce.default
    )


def is_all_to_all_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_to_all_single.default
    )


# Helper functions moved to top for better organization
def _ag_group_key(node: torch.fx.Node) -> tuple[str, torch.dtype]:  # type: ignore[name-defined]
    _, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    assert isinstance(group_name, str)
    return (group_name, dtype)


def _ag_group_key_multidtype(node: torch.fx.Node) -> tuple[str]:
    _, group_size, group_name = node.args
    assert isinstance(group_name, str)
    return (group_name,)


def _rs_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:  # type: ignore[name-defined]
    _, reduce_op, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    assert isinstance(group_name, str)
    assert isinstance(reduce_op, str)
    return (group_name, reduce_op, dtype)


def _ar_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
    _, reduce_op, group_name = node.args
    dtype = node.meta["val"].dtype
    assert isinstance(group_name, str)
    assert isinstance(reduce_op, str)
    return (group_name, reduce_op, dtype)


def bucket_key(node: torch.fx.Node, mode: BucketMode | None = None) -> object | None:
    if is_all_gather_into_tensor(node):
        group_key_fn = (
            _ag_group_key_multidtype if mode and "multidtype" in mode else _ag_group_key
        )
        return group_key_fn(node)
    elif is_reduce_scatter_tensor(node):
        return _rs_group_key(node)
    elif is_all_reduce_tensor(node):
        return _ar_group_key(node)
    else:
        return None


def has_mergeable_all_gather_convert_dtype(n: torch.fx.Node) -> bool:
    node_in = n.args[0]
    return (
        is_all_gather_into_tensor(n)
        and isinstance(node_in, torch.fx.Node)
        and node_in.op == "call_function"
        and (
            node_in.target is torch.ops.prims.convert_element_type.default
            or node_in.target is torch.ops.aten._to_copy.default
        )
        and len(node_in.users) == 1
    )


def pick_bucket_dtype(dtypes: list[torch.dtype]) -> torch.dtype:  # type: ignore[name-defined]
    assert len(dtypes) > 0
    return min(dtypes, key=operator.attrgetter("itemsize"))


@torch.library.custom_op("bucketing::_pre_bucket_reduce_scatter", mutates_args={})
def _pre_bucket_reduce_scatter(
    rs_ins: list[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    rs_ins_flattened = [x.view(group_size, -1) for x in rs_ins]
    new_rs_in = torch.cat(rs_ins_flattened, dim=1).flatten()
    return new_rs_in


def _pre_bucket_reduce_scatter_fake(
    rs_ins: list[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    out_numel = sum(rs_in.numel() for rs_in in rs_ins)
    return torch.empty((out_numel,), device=rs_ins[0].device, dtype=rs_ins[0].dtype)


_pre_bucket_reduce_scatter.register_fake(_pre_bucket_reduce_scatter_fake)


# List of all torch dtypes for serialization through custom ops
# TODO: custom ops support list[dtype] input
_ALL_DTYPES = tuple(
    [
        getattr(torch, attr)
        for attr in dir(torch)
        if isinstance(getattr(torch, attr), torch.dtype)
    ]
)


@torch.library.custom_op("bucketing::_pre_bucket_all_gather", mutates_args={})
def _pre_bucket_all_gather(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtype_ints: list[
        int
    ],  # dtype enum values, that inputs are converted to before all_gather
    rank: int,
) -> torch.Tensor:
    # Convert int indices back to torch.dtype
    out_dtypes = [_ALL_DTYPES[d] for d in out_dtype_ints]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes, strict=True)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    foreach_copy_dsts = torch.split(new_ag_in, ins_split_sizes)
    # View each destination slice as its output dtype, then copy
    # The copy operation handles dtype conversion from input dtype to output dtype
    foreach_copy_dsts_typed = [
        dst.view(out_dtype)
        for dst, out_dtype in zip(foreach_copy_dsts, out_dtypes, strict=True)
    ]
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    torch._foreach_copy_(foreach_copy_dsts_typed, ag_ins_flattened)
    return new_ag_out


def _pre_bucket_all_gather_fake(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    out_dtype_ints: list[int],
    rank: int,
) -> torch.Tensor:
    out_dtypes = [_ALL_DTYPES[d] for d in out_dtype_ints]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes, strict=True)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    return new_ag_out


_pre_bucket_all_gather.register_fake(_pre_bucket_all_gather_fake)


def _move_nodes_after_wait_insertion_point(new_nodes: list[fx.Node], wait_insertion_point: fx.Node) -> None:
    # Find the first wait node in new_nodes
    wait_start_idx = None
    for i, node in enumerate(new_nodes):
        if is_wait_tensor(node):
            wait_start_idx = i
            break

    # Move all nodes from wait onwards (including the wait)
    if wait_start_idx is not None:
        nodes_to_move = new_nodes[wait_start_idx:]
        for node in nodes_to_move:
            wait_insertion_point.prepend(node)


def _erase_original_nodes(g: fx.Graph, op_nodes: list[fx.Node], _ag_pre_nodes: Optional[dict[fx.Node, list[fx.Node]]] = None) -> None:
    ag_pre_nodes = _ag_pre_nodes or defaultdict(list)

    for n in op_nodes:
        assert len(n.users) == 1, f"Expected single user(wait) for {n}, got {len(n.users)}"
        wait_n = next(iter(n.users))
        g.erase_node(wait_n)
        g.erase_node(n)
        for pre in reversed(ag_pre_nodes[n]):
            g.erase_node(pre)


def merge_all_gather_bucket(
    g: torch.fx.Graph,
    ag_nodes: list[torch.fx.Node],
    insert_before: torch.fx.Node | None = None,
    wait_insertion_point: torch.fx.Node | None = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    from torch.distributed.distributed_c10d import _resolve_process_group

    ag0 = ag_nodes[0]
    _, group_size, group_name = ag0.args
    bucket_ins: list[fx.Node] = []
    ag_pre_nodes: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    out_dtypes: list[torch.dtype] = []
    ag_ins = []
    for n in ag_nodes:
        assert n.args[1] == group_size and n.args[2] == group_name
        out_dtypes.append(n.meta["val"].dtype)

        node_in = n.args[0]
        if has_mergeable_all_gather_convert_dtype(n):
            ag_pre_nodes[n].append(node_in)
            node_in = node_in.args[0]  # type: ignore[assignment]
        assert isinstance(node_in, fx.Node)
        bucket_ins.append(node_in)
        ag_ins.append(node_in.meta["val"])

    bucket_dtype = pick_bucket_dtype(out_dtypes)

    # Process bucket with lazy input collection
    rank: int = dist.get_rank(_resolve_process_group(group_name))

    out_dtype_ints = [_ALL_DTYPES.index(dt) for dt in out_dtypes]

    new_nodes: list[fx.Node] = []
    replacements: dict[fx.Node, fx.Node] = {}
    wait_to_start: dict[fx.Node, fx.Node] = {}

    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes)
    ]
    bucket_dtype_size_bytes = bucket_dtype.itemsize
    ins_split_sizes = [b // bucket_dtype_size_bytes for b in ins_split_sizes_bytes]
    ag_input_numel = sum(ins_split_sizes)
    with g.inserting_before(insert_before):
        new_ag_out = g.call_function(torch.ops.bucketing._pre_bucket_all_gather,
            args=(bucket_ins, group_size, group_name, bucket_dtype, out_dtype_ints, rank),
            kwargs={},)
        new_nodes.append(new_ag_out)

        new_ag_in = g.call_function(
            torch.ops.aten.narrow,
            args=(new_ag_out, 0, ag_input_numel*rank, ag_input_numel),
            kwargs={},
        )
        new_nodes.append(new_ag_in)

        ag_start = g.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor_out.default,
            args=(new_ag_in, group_size, group_name),
            kwargs={"out": new_ag_out},
        )
        ag_start.meta["manual_bucket_node_type"] = "bucketed_all_gather"
        new_nodes.append(ag_start)

        ag_wait = g.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(ag_start,),
            kwargs={},
        )
        ag_wait.meta["manual_bucket_node_type"] = "bucketed_all_gather_wait"
        new_nodes.append(ag_wait)
        wait_to_start[ag_wait] = ag_start

        viewed = g.call_function(torch.ops.aten.view.default, args=(ag_wait, (group_size, -1)), kwargs={})
        new_nodes.append(viewed)

        outs_bucket_dtype = g.call_function(
            torch.ops.aten.split_with_sizes.default,
            args=(viewed, ins_split_sizes, 1),
            kwargs={},
        )
        new_nodes.append(outs_bucket_dtype)
        for i, (ag_in, out_dt) in enumerate(zip(ag_ins, out_dtypes, strict=True)):
            shape = ag_in.shape
            o_i = g.call_function(operator.getitem, args=(outs_bucket_dtype, i), kwargs={})
            new_nodes.append(o_i)

            viewed_dtype = g.call_function(torch.ops.aten.view.dtype, args=(o_i, out_dt), kwargs={})
            new_nodes.append(viewed_dtype)

            new_shape = (shape[0] * group_size,) + tuple(shape[1:])
            reshaped = g.call_function(torch.ops.aten.reshape.default, args=(viewed_dtype, new_shape), kwargs={})
            new_nodes.append(reshaped)

            old_wait = next(iter(ag_nodes[i].users))
            replacements[old_wait] = reshaped

    if wait_insertion_point is not None:
        _move_nodes_after_wait_insertion_point(new_nodes, wait_insertion_point)

    for old_wait, new_out in replacements.items():
        old_wait.replace_all_uses_with(new_out)

    _erase_original_nodes(g, ag_nodes, ag_pre_nodes)
    return new_nodes, replacements


def merge_reduce_scatter_bucket(
    g: torch.fx.Graph,
    rs_nodes: list[torch.fx.Node],
    insert_before: torch.fx.Node | None = None,
    wait_insertion_point: torch.fx.Node | None = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    # Validate bucket consistency
    rs0 = rs_nodes[0]
    _, reduce_op, group_size, group_name = rs0.args
    assert isinstance(group_name, str) and isinstance(reduce_op, str)

    bucket_ins: list[fx.Node] = []
    out_sizes = []
    out_numels = []
    for n in rs_nodes:
        assert n.args[1] == reduce_op and n.args[2] == group_size and n.args[3] == group_name
        x = n.args[0]
        assert isinstance(x, fx.Node)
        bucket_ins.append(x)

        # trace fn: new_out_sizes = (x.shape[0]//group_size,) + x.shape[1:]
        x_val = x.meta["val"]
        out_sizes.append((x_val.shape[0] // group_size,) + tuple(x_val.shape[1:]))
        out_numels.append(x_val.numel() // group_size)

    new_nodes: list[fx.Node] = []
    replacements: dict[fx.Node, fx.Node] = {}

    with g.inserting_before(insert_before):
        new_rs_in = g.call_function(
            torch.ops.bucketing._pre_bucket_reduce_scatter,
            args=(bucket_ins, group_size),
            kwargs={},
        )
        new_nodes.append(new_rs_in)

        rs_start = g.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(new_rs_in, reduce_op, group_size, group_name),
            kwargs={},
        )
        rs_start.meta["manual_bucket_node_type"] = "bucketed_reduce_scatter"
        new_nodes.append(rs_start)

        rs_wait = g.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(rs_start,),
            kwargs={},
        )
        rs_wait.meta["manual_bucket_node_type"] = "bucketed_reduce_scatter_wait"
        new_nodes.append(rs_wait)

        # split outputs
        outs_flat = g.call_function(torch.ops.aten.split_with_sizes.default, args=(rs_wait, out_numels, 0), kwargs={})
        new_nodes.append(outs_flat)

        for i, s in enumerate(out_sizes):
            o_i = g.call_function(operator.getitem, args=(outs_flat, i), kwargs={})
            new_nodes.append(o_i)
            viewed = g.call_function(torch.ops.aten.view.default, args=(o_i, s), kwargs={})
            new_nodes.append(viewed)

            old_wait = next(iter(rs_nodes[i].users))
            replacements[old_wait] = viewed

    if wait_insertion_point is not None:
        _move_nodes_after_wait_insertion_point(new_nodes, wait_insertion_point)

    for old_wait, new_out in replacements.items():
        old_wait.replace_all_uses_with(new_out)

    _erase_original_nodes(g, rs_nodes)
    return new_nodes, replacements
