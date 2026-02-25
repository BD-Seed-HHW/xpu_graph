import copy
import itertools
import os
from typing import Callable

import torch
from torch.fx import GraphModule
from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode

from xpu_graph.config import get_dump_dir
from xpu_graph.utils import logger

CASE_CNT = itertools.count()


def intercept(target_fn, *, golden_fn, fw_metadata, is_training, mark=0, config_str=""):
    check_configs = {}
    use_golden = False
    for item in config_str.split(","):
        key, val = item.split("=")
        if key in ["rtol", "atol"]:
            val = float(val)
            check_configs[key] = val
        elif key in ["allow_subclasses", "equal_nan", "check_device", "check_dtype", "check_layout", "check_stride"]:
            val = val.lower() in ["true", "1", "on"]
            check_configs[key] = val
        elif key == "use_golden":
            use_golden = val.lower() in ["true", "1", "on"]
        else:
            logger.warning(f"Invalid key: {key}")

    impure = fw_metadata is None or fw_metadata.num_mutated_inp_runtime_indices > 0

    if is_training:
        return AutogradInterceptor(golden_fn, mark, use_golden=use_golden, impure=impure, **check_configs).guard(
            target_fn
        )
    else:
        return FunctionInterceptor(golden_fn, mark, use_golden=use_golden, impure=impure, **check_configs).guard(
            target_fn
        )


def _invoke_inference(func, inputs):
    outputs = func(*inputs)
    return outputs


def _invoke_forward(func, inputs):
    with torch.enable_grad():
        outputs = func(*inputs)
    return outputs


def _invoke_backward(func, inputs, outputs, grad_outputs, inputs_requires_grad):
    inputs_ad_idx = []
    inputs_ad = []
    grad_outputs_ad = []
    outputs_ad = []
    for i, t in enumerate(inputs):
        if inputs_requires_grad[i]:
            inputs_ad_idx.append(i)
            inputs_ad.append(t)
        elif isinstance(t, torch.Tensor) and t.requires_grad:
            grad_outputs_ad.append(grad_outputs[i])
            outputs_ad.append(t)

    for i, t in enumerate(outputs):
        if isinstance(t, torch.Tensor) and t.requires_grad:
            grad_outputs_ad.append(grad_outputs[i + len(inputs)])
            outputs_ad.append(t)

    grad_inputs_ad = torch.autograd.grad(outputs_ad, inputs_ad, grad_outputs_ad, allow_unused=True)

    grad_inputs = [None] * len(inputs)
    for i, g in zip(inputs_ad_idx, grad_inputs_ad):
        grad_inputs[i] = g

    return tuple(grad_inputs)


def compare_tensor_list(base_list, trgt_list, **check_configs):
    diverge_cnt = 0
    for i, (base_t, trgt_t) in enumerate(zip(base_list, trgt_list)):
        try:
            torch.testing.assert_close(trgt_t, base_t, **check_configs)
        except AssertionError as e:
            diverge_cnt += 1
            logger.warning(f"Tensor {i} diverges:\n{e}")
    return diverge_cnt


class FunctionInterceptor:
    def __init__(self, golden_fn: Callable, mark=0, use_golden=False, impure=True, **check_configs):
        # Note: The monitor is used for inference.
        # We suppose that original gm has no states (as torch dynamo does, which treats parameters as inputs)
        self.golden_fn = golden_fn
        self.mark = mark
        self.use_golden = use_golden
        self.impure = impure
        self.check_configs = check_configs

    def guard(self, compiled_fn):
        ## base func is the actual func in the original autograd graph
        ## trgt func is executed alongside to compare with
        if self.use_golden:
            base_func, trgt_func = self.golden_fn, compiled_fn
        else:
            base_func, trgt_func = compiled_fn, self.golden_fn

        # Similar to what torch._functorch.autograd does, wrap the forward function again
        def monitored_inference(*inputs):
            logger.info("Monitored inference")

            base_inputs = inputs
            if self.impure:
                backup_inputs = copy.deepcopy(base_inputs)
            else:
                backup_inputs = base_inputs

            # restore the random state after golden forward
            if self.impure:
                trgt_inputs = copy.deepcopy(base_inputs)
            else:
                trgt_inputs = base_inputs

            # execute the trgt func first to restore rng state
            if (device_type := torch._utils._get_available_device_type()) is not None:
                with torch.random.fork_rng(device_type=device_type):
                    trgt_outputs = _invoke_inference(trgt_func, trgt_inputs)
            else:
                rng_state = torch.random.get_rng_state()
                trgt_outputs = _invoke_inference(trgt_func, trgt_inputs)
                torch.random.set_rng_state(rng_state)

            base_outputs = _invoke_inference(base_func, base_inputs)
            input_diverge_cnt = 0
            if self.impure:
                input_diverge_cnt = compare_tensor_list(base_inputs, trgt_inputs, **self.check_configs)
            output_diverge_cnt = compare_tensor_list(base_outputs, trgt_outputs, **self.check_configs)
            diverge_cnt = input_diverge_cnt + output_diverge_cnt
            if diverge_cnt > 0:
                global CASE_CNT
                case_id = next(CASE_CNT)
                dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_inference_{case_id}")
                logger.warning(
                    f"Summary: The inference pass diverges for {self.golden_fn}\n"
                    f"         input_diverge_cnt: {input_diverge_cnt}, output_diverge_cnt: {output_diverge_cnt}\n"
                    f"         cases saved_to: {dump_path}"
                )
                os.makedirs(dump_path, exist_ok=True)
                if isinstance(self.golden_fn, GraphModule):
                    with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                        mod_str = self.golden_fn.print_readable(
                            print_output=False, include_stride=True, include_device=True
                        )
                        gm_f.write(mod_str)
                if self.use_golden:
                    golden_inputs, actual_inputs = base_inputs, trgt_inputs
                    golden_outputs, actual_outputs = base_outputs, trgt_outputs
                else:
                    golden_inputs, actual_inputs = trgt_inputs, base_inputs
                    golden_outputs, actual_outputs = trgt_outputs, base_outputs
                dump_glob = {
                    "inputs": backup_inputs,
                    "golden_inputs": golden_inputs,
                    "actual_inputs": actual_inputs,
                    "golden_outputs": golden_outputs,
                    "actual_outputs": actual_outputs,
                }
                torch.save(
                    dump_glob,
                    os.path.join(dump_path, "dump_glob_inf.pt"),
                )

            return base_outputs

        if hasattr(compiled_fn, "zero_grad"):
            monitored_inference.zero_grad = compiled_fn.zero_grad
        if hasattr(compiled_fn, "named_parameters"):
            monitored_inference.named_parameters = compiled_fn.named_parameters
        if hasattr(compiled_fn, "named_buffers"):
            monitored_inference.named_buffers = compiled_fn.named_buffers
        return monitored_inference


class AutogradInterceptor:
    def __init__(self, golden_fn: Callable, mark=0, use_golden=False, impure=True, **check_configs):
        # Note: The monitor is used for training.
        # We suppose that original gm has no states (as torch dynamo does, which treats parameters as inputs)
        self.golden_fn = golden_fn
        if isinstance(golden_fn, Module):
            assert len(golden_fn.state_dict()) == 0
        self.mark = mark
        self.use_golden = use_golden
        self.impure = impure
        self.check_configs = check_configs

    def guard(self, compiled_fn):
        if self.use_golden:
            base_func = self.golden_fn
            trgt_func = compiled_fn
        else:
            base_func = compiled_fn
            trgt_func = self.golden_fn

        class MonitoredCompiled(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                logger.info("Monitored forward")

                inputs_requires_grad = [isinstance(t, torch.Tensor) and t.requires_grad for t in inputs]
                base_inputs = [
                    t.detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t for t in inputs
                ]

                if self.impure:
                    backup_inputs = copy.deepcopy(base_inputs)
                else:
                    backup_inputs = base_inputs

                # restore the random state after golden forward
                if self.impure:
                    trgt_inputs = copy.deepcopy(base_inputs)
                else:
                    trgt_inputs = base_inputs
                if (device_type := torch._utils._get_available_device_type()) is not None:
                    with torch.random.fork_rng(device_type=device_type):
                        trgt_outputs = _invoke_forward(trgt_func, trgt_inputs)
                else:
                    rng_state = torch.random.get_rng_state()
                    trgt_outputs = _invoke_forward(trgt_func, trgt_inputs)
                    torch.random.set_rng_state(rng_state)

                base_outputs = _invoke_forward(base_func, base_inputs)
                input_diverge_cnt = 0
                if self.impure:
                    input_diverge_cnt = compare_tensor_list(base_inputs, trgt_inputs, **self.check_configs)
                output_diverge_cnt = compare_tensor_list(base_outputs, trgt_outputs, **self.check_configs)
                diverge_cnt = input_diverge_cnt + output_diverge_cnt
                if diverge_cnt > 0:
                    global CASE_CNT
                    case_id = next(CASE_CNT)
                    dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_forward_{case_id}")
                    logger.warning(
                        f"Summary: The forward pass diverges for {self.golden_fn}\n"
                        f"         input_diverge_cnt: {input_diverge_cnt}, output_diverge_cnt: {output_diverge_cnt}\n"
                        f"         cases saved_to: {dump_path}"
                    )
                    os.makedirs(dump_path, exist_ok=True)
                    if isinstance(self.golden_fn, GraphModule):
                        with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                            mod_str = self.golden_fn.print_readable(
                                print_output=False, include_stride=True, include_device=True
                            )
                            gm_f.write(mod_str)

                    if self.use_golden:
                        golden_inputs, actual_inputs = base_inputs, trgt_inputs
                        golden_outputs, actual_outputs = base_outputs, trgt_outputs
                    else:
                        golden_inputs, actual_inputs = trgt_inputs, base_inputs
                        golden_outputs, actual_outputs = trgt_outputs, base_outputs

                    dump_glob = {
                        "inputs": backup_inputs,
                        "golden_inputs": golden_inputs,
                        "actual_inputs": actual_inputs,
                        "golden_outputs": golden_outputs,
                        "actual_outputs": actual_outputs,
                    }
                    torch.save(
                        dump_glob,
                        os.path.join(dump_path, "dump_glob_fwd.pt"),
                    )

                ctx.saved_states = {
                    "base_inputs": base_inputs,
                    "trgt_inputs": trgt_inputs,
                    "base_outputs": base_outputs,
                    "trgt_outputs": trgt_outputs,
                    "inputs_requires_grad": inputs_requires_grad,
                }
                inputs_after = [
                    (
                        t_after.detach().requires_grad_(True)
                        if isinstance(t_after, torch.Tensor) and t_after.requires_grad and not req_grad
                        else None
                    )
                    for t_after, req_grad in zip(base_inputs, inputs_requires_grad)
                ]
                outputs = [
                    t.detach().requires_grad_(True) if isinstance(t, torch.Tensor) and t.requires_grad else t
                    for t in base_outputs
                ]
                # Note: wrap the results with a tuple, thus the backward function is guaranteed to be triggered
                return tuple(inputs_after + outputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                logger.info("Monitored backward")

                base_inputs = ctx.saved_states["base_inputs"]
                trgt_inputs = ctx.saved_states["trgt_inputs"]
                base_outputs = ctx.saved_states["base_outputs"]
                trgt_outputs = ctx.saved_states["trgt_outputs"]
                inputs_requires_grad = ctx.saved_states["inputs_requires_grad"]

                trgt_grad_inputs = _invoke_backward(
                    trgt_func, trgt_inputs, trgt_outputs, grad_outputs, inputs_requires_grad
                )
                base_grad_inputs = _invoke_backward(
                    base_func, base_inputs, base_outputs, grad_outputs, inputs_requires_grad
                )
                grad_diverge_cnt = compare_tensor_list(base_grad_inputs, trgt_grad_inputs, **self.check_configs)
                if grad_diverge_cnt > 0:
                    global CASE_CNT
                    case_id = next(CASE_CNT)
                    dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_backward_{case_id}")
                    logger.warning(
                        f"Summary: The backward pass diverges for {self.golden_fn}\n"
                        f"         grad_diverge_cnt: {grad_diverge_cnt}\n"
                        f"         cases saved_to: {dump_path}"
                    )
                    os.makedirs(dump_path, exist_ok=True)
                    if isinstance(self.golden_fn, GraphModule):
                        with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                            mod_str = self.golden_fn.print_readable(
                                print_output=False, include_stride=True, include_device=True
                            )
                            gm_f.write(mod_str)
                    if self.use_golden:
                        golden_inputs, actual_inputs = base_inputs, trgt_inputs
                        golden_grad_inputs, actual_grad_inputs = base_grad_inputs, trgt_grad_inputs
                    else:
                        golden_inputs, actual_inputs = trgt_inputs, base_inputs
                        golden_grad_inputs, actual_grad_inputs = trgt_grad_inputs, base_grad_inputs
                    dump_glob = {
                        "golden_inputs": golden_inputs,
                        "actual_inputs": actual_inputs,
                        "grad_outputs": grad_outputs,
                        "golden_grad_inputs": golden_grad_inputs,
                        "actual_grad_inputs": actual_grad_inputs,
                    }
                    torch.save(
                        dump_glob,
                        os.path.join(dump_path, "dump_glob_bwd.pt"),
                    )
                return base_grad_inputs

        # Similar to what torch._functorch.autograd does, wrap the forward function again
        def monitored_forward(*inputs):
            outputs = MonitoredCompiled.apply(*inputs)
            after_inputs = outputs[: len(inputs)]
            for t, t_after in zip(inputs, after_inputs):
                # filter out inplace mutations with grad
                if t_after is not None:
                    t.copy_(t_after)

            outputs = outputs[len(inputs) :]
            return outputs

        if hasattr(compiled_fn, "zero_grad"):
            monitored_forward.zero_grad = compiled_fn.zero_grad
        if hasattr(compiled_fn, "named_parameters"):
            monitored_forward.named_parameters = compiled_fn.named_parameters
        if hasattr(compiled_fn, "named_buffers"):
            monitored_forward.named_buffers = compiled_fn.named_buffers
        return monitored_forward


class OpInterceptor(TorchDispatchMode):
    def __init__(self, golden_funcs, mark=None, dispatch_key=None, **check_configs):
        super().__init__(dispatch_key)
        self.golden_funcs = golden_funcs
        self.mark = mark if mark is not None else "op"
        self.check_configs = check_configs

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if func in self.golden_funcs:
            golden_fn = self.golden_funcs[func]
            logger.info(f"Monitored op: {func}")

            actual_inputs = args
            golden_inputs = copy.deepcopy(actual_inputs)
            if (device_type := torch._utils._get_available_device_type()) is not None:
                with torch.random.fork_rng(device_type=device_type):
                    golden_outputs = golden_fn(*golden_inputs, **(kwargs or {}))
            else:
                rng_state = torch.random.get_rng_state()
                golden_outputs = golden_fn(*golden_inputs, **(kwargs or {}))
                torch.random.set_rng_state(rng_state)
            actual_outputs = func(*args, **(kwargs or {}))
            input_diverge_cnt = compare_tensor_list(actual_inputs, golden_inputs, **self.check_configs)
            output_diverge_cnt = compare_tensor_list(actual_outputs, golden_outputs, **self.check_configs)
            if input_diverge_cnt > 0 or output_diverge_cnt > 0:
                global CASE_CNT
                case_id = next(CASE_CNT)
                dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_op_{case_id}")
                logger.warning(
                    f"Summary: The op diverges for {func}\n"
                    f"         input_diverge_cnt: {input_diverge_cnt}, output_diverge_cnt: {output_diverge_cnt}\n"
                    f"         cases saved_to: {dump_path}"
                )
                os.makedirs(dump_path, exist_ok=True)
                dump_glob = {
                    "golden_inputs": golden_inputs,
                    "actual_inputs": actual_inputs,
                    "golden_outputs": golden_outputs,
                    "actual_outputs": actual_outputs,
                }
                torch.save(
                    dump_glob,
                    os.path.join(dump_path, "dump_glob_op.pt"),
                )
            return actual_outputs
        return func(*args, **(kwargs or {}))
