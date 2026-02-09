import os, sys
from dataclasses import dataclass
from typing import Callable

import torch
from parallel_dims import ParallelizeDims
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3ToyConfig
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from tests.npu.test_dist_utils import dist_setup, cleanup
from xpu_graph.compiler import XpuGraph
from xpu_graph.config import OptLevel, Target, XpuGraphConfig

from xpu_graph.utils import logger


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


def compile_model(model: Qwen3ForCausalLM):
    module_bucket_plans = get_transformer_block_buckets(model)
    logger.info(f"module_bucket_plans: {module_bucket_plans}")
    xpu_graph_backend = XpuGraph(
        XpuGraphConfig(
            is_training=True,
            freeze=False,
            target=Target.npu,
            opt_level=OptLevel.level1,
            debug=True,
            overlap_manual_scheduling=False,
            vendor_compiler_config=None,
        ),
        module_bucket_plans=module_bucket_plans,
    )
    model = torch.compile(model, backend=xpu_graph_backend)
    logger.info(f"compile model successfully")
    return model


def forward_backward_step(model, optimizer, loss_fn, data, target):
    optimizer.zero_grad()
    logits = model(data)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target[:, 1:].contiguous()

    B, Tm1, V = shift_logits.shape
    loss = loss_fn(
        shift_logits.view(B * Tm1, V),
        shift_labels.view(B * Tm1),
    )

    loss.backward()
    optimizer.step()
    return loss


def train(rank, train_config):
    rank = rank
    model = Qwen3ForCausalLM(train_config.model_config)
    model.load_state_dict(torch.load(train_config.model_path))
    if train_config.parallelize_dims.world_size > 1:
        dist_setup(rank, train_config.parallelize_dims.world_size)
        train_config.parallelize_fn(model, train_config.parallelize_dims)
        if train_config.is_compile:
            model = compile_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    loss_fn = train_config.loss_fn
    mini_batch_size = train_config.batch_size // dist.get_world_size() if dist.is_initialized() else train_config.batch_size
    data_loader = get_dataloader(batch_size=mini_batch_size, 
                                        num_samples=train_config.num_samples,
                                        shuffle=train_config.shuffle,
                                        path=train_config.dataset_path)
    model.train().to(train_config.device)
    torch.set_grad_enabled(True)
    global_step = 0
    for epoch in range(train_config.epochs):
        total_loss = 0.0
        n_batch = 0
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(data_loader):
            global_step += 1
            data, target = data.to(train_config.device), target.to(train_config.device)
            loss = forward_backward_step(model, optimizer, loss_fn, data, target)
            total_loss += loss
            # if global_step % 5 == 0:
            logger.info(f"rank[{rank}]: Epoch [{epoch}], Step [{global_step}], Loss: {loss:.4f}")
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
    folder = "/tmp/test"
    if dist.is_initialized():
        if not os.path.exists(folder) and rank == 0:
            os.makedirs(folder)
        save_dist_model(model, f"{folder}/fsdp.pt")
        logger.info(f"Full state dict saved to {folder}/fsdp.pt")
    else:
        if not os.path.exists(folder):
            os.makedirs(folder)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(sd, f"{folder}/no_fsdp.pt")
        logger.info(f"Full state dict saved to {folder}/no_fsdp.pt")
    if dist.is_initialized():
        dist.barrier()
        cleanup()


def save_dist_model(model, save_path: str):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    cpu_state_dict = get_model_state_dict(model, options=options)
    if dist.get_rank() == 0:
        torch.save(cpu_state_dict, save_path)
