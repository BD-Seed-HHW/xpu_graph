import datetime
import os
from dataclasses import dataclass
from functools import reduce
from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from xpu_graph.compiler import XpuGraph
from xpu_graph.config import XpuGraphConfig
from xpu_graph.utils import logger, setup_logger

from tests.npu.test_dist_utils import cleanup, dist_setup

from .modeling_qwen3 import Qwen3ForCausalLM, Qwen3ToyConfig
from .parallel_dims import ParallelizeDims


def compute_tensor_xor(tensor: torch.Tensor) -> int:
    int8_data = tensor.detach().cpu().view(torch.int8).flatten()
    if isinstance(int8_data, torch.distributed._tensor.DTensor):
        int8_data = int8_data.to_local()
    xor_result = reduce(lambda x, y: x ^ y, int8_data.tolist(), 0)
    return xor_result


def compute_grad_xors(model) -> dict:
    grad_xors = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_xors[name] = compute_tensor_xor(param.grad)
    return grad_xors


@dataclass
class TrainConfig:
    # training setting
    is_training: bool = True
    epochs: int = 5 # Number of training epochs
    steps: int = 200 # Number of training steps
    batch_size: int = 24 # Batch size
    dataset: str = "" # Path to the training data
    device: str = "npu" # Device to use for training
    model_path: str = "/opt/tiger/Qwen3-0.6B" # Path to the model to train
    dataset_path: str = "/tmp/model/data.pt" # Path to the training data
    shuffle: bool = False # Whether to shuffle the training data
    max_seq_len: int = 1024 # Maximum sequence length
    model_config: Qwen3ToyConfig = None

    is_compile: bool = False # Whether to compile the model

    num_samples: int = 4096 # Number of samples in the training dataset

    parallelize_dims: ParallelizeDims = None # Parallelize dimensions
    parallelize_fn: Callable = None # Parallelize function

    # debug setting
    is_debug: bool = False

    # lr scheduler
    warmup_steps: int = 100 # Number of warmup steps
    lr: float = 1 # Learning rate, because we use random dataset and random weight, so the lr must be a little bigger

    loss_fn: Callable = None # Loss function to use for training
    seed : int = 111


class SimpleDataset(Dataset):
    def __init__(self, path: str, num_samples: int):
        self.len = num_samples
        self.data = torch.load(path)

    def __getitem__(self, index):
        return self.data[index % self.data.shape[0]], self.data[index % self.data.shape[0]]

    def __len__(self):
        return self.len


def get_dataloader(batch_size, num_samples, shuffle=False, path: str = None):
    dataset = SimpleDataset(path, num_samples)
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=0,  # 简单起见，不用多进程
        pin_memory=True,  # 加速 CPU->GPU 传输
        drop_last=True,
    )

## only support model written in transformer-like style, just like Qwen3ForCausalLM->Qwen3Model->Qwen3DecoderLayer...
def get_transformer_block_buckets(model: Qwen3ForCausalLM) -> list[list[str] | str]:
    module_list = []
    # module_list.append(model.model.embed_tokens)
    for transformer_block in model.model.layers:
        module_list.append(transformer_block)
    # module_list.append([model.model.norm, model.lm_head])

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                if fqn_list := convert_modules_to_fqns(m, module_to_fqn_mapping):
                    result.append(fqn_list)
            else:
                if fqn := module_to_fqn_mapping.get(m):
                    result.append(fqn)
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def compile_model(model: Qwen3ForCausalLM, xpu_graph_config: XpuGraphConfig):
    module_bucket_plans = get_transformer_block_buckets(model)
    logger.info(f"module_bucket_plans: {module_bucket_plans}")
    xpu_graph_backend = XpuGraph(
        xpu_graph_config,
        module_bucket_plans=module_bucket_plans,
    )
    model = torch.compile(model, backend=xpu_graph_backend)
    logger.info("compile model successfully")
    return model


def forward_backward_step(model, optimizer, loss_fn, data, target, rank):
    optimizer.zero_grad()
    time0 = datetime.datetime.now()
    event0 = torch.npu.Event(enable_timing=True)
    event0.record()
    logits = model(data)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target[:, 1:].contiguous()

    B, Tm1, V = shift_logits.shape
    loss = loss_fn(
        shift_logits.view(B * Tm1, V),
        shift_labels.view(B * Tm1),
    )

    loss.backward()

    event1 = torch.npu.Event(enable_timing=True)
    event1.record()
    time1 = datetime.datetime.now()
    torch.npu.synchronize()
    time2 = datetime.datetime.now()
    host_time_step = (time1 - time0).total_seconds()
    npu_time_step = event0.elapsed_time(event1) / 1000.0
    e2e_time_step = (time2 - time0).total_seconds()

    grad_xors = compute_grad_xors(model)

    optimizer.step()

    return loss.detach(), grad_xors, host_time_step, npu_time_step, e2e_time_step


def train(rank, train_config, path, xpu_graph_config):
    setup_logger(is_debug=True)
    rank = rank
    model = Qwen3ForCausalLM(train_config.model_config)
    model.load_state_dict(torch.load(train_config.model_path))
    if train_config.parallelize_dims.world_size > 1:
        dist_setup(rank, train_config.parallelize_dims.world_size)
        train_config.parallelize_fn(model, train_config.parallelize_dims)
    if train_config.is_compile:
        model = compile_model(model, xpu_graph_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    loss_fn = train_config.loss_fn
    mini_batch_size = train_config.batch_size // dist.get_world_size() if dist.is_initialized() else train_config.batch_size
    data_loader = get_dataloader(batch_size=mini_batch_size,
                                        num_samples=train_config.num_samples,
                                        shuffle=train_config.shuffle,
                                        path=train_config.dataset_path)
    model.train().to(train_config.device)
    torch.set_grad_enabled(True)
    global global_step
    global_step = 0

    all_grad_xors = {}
    host_time, npu_time, e2e_time = 0.0, 0.0, 0.0

    for epoch in range(train_config.epochs):
        total_loss = 0.0
        n_batch = 0
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(data_loader):
            global_step += 1
            data, target = data.to(train_config.device), target.to(train_config.device)

            loss, grad_xors, host_time_step, npu_time_step, e2e_time_step = \
                forward_backward_step(model, optimizer, loss_fn, data, target, rank)
            host_time += host_time_step
            e2e_time += e2e_time_step
            npu_time += npu_time_step
            all_grad_xors[global_step] = grad_xors
            total_loss += loss
            logger.info(f"rank[{rank}]: Epoch [{epoch}], Step [{global_step}], Loss: {loss:.4f}, "
                        f"Host time: {host_time_step:.4f}, NPU time: {npu_time_step:.4f}, E2E time: {e2e_time_step:.4f}")
            n_batch += 1
            if global_step >= train_config.steps:
                break
        total_loss = total_loss / n_batch
        if dist.is_initialized():
            dist.all_reduce(total_loss, dist.ReduceOp.SUM)
            total_loss = total_loss / dist.get_world_size()
        logger.info(f"rank[{rank}]: Epoch [{epoch}], Loss: {total_loss:.4f}")
        if global_step >= train_config.steps:
            break

    logger.info(f"rank[{rank}]: Host time: {host_time / global_step:.4f}/step")
    logger.info(f"rank[{rank}]: NPU time: {npu_time / global_step:.4f}/step")
    logger.info(f"rank[{rank}]: E2E time: {e2e_time / global_step:.4f}/step")

    folder = os.path.dirname(path)
    filename = os.path.basename(path)
    if dist.is_initialized():
        if not os.path.exists(folder) and rank == 0:
            os.makedirs(folder)
        dist.barrier()
        save_dist_model(model, path)
        grad_xor_path = os.path.join(folder, f"{filename.split('.')[0]}_grad_rank{rank}.pt")
        torch.save(all_grad_xors, grad_xor_path)
        logger.info(f"rank[{rank}]: Grad xors saved to {grad_xor_path}")

        if rank == 0:
            logger.info(f"Full state dict saved to {path}")
    else:
        if not os.path.exists(folder):
            os.makedirs(folder)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(sd, path)
        grad_xor_path = os.path.join(folder, "grad_xors.pt")
        torch.save(all_grad_xors, grad_xor_path)
        logger.info(f"Full state dict saved to {path}")
        logger.info(f"Grad xors saved to {grad_xor_path}")
    if dist.is_initialized():
        dist.barrier()
        cleanup()


def save_dist_model(model, save_path: str):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    cpu_state_dict = get_model_state_dict(model, options=options)
    if dist.get_rank() == 0:
        torch.save(cpu_state_dict, save_path)
