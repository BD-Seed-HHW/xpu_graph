import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


class MainModel(nn.Module):
    def __init__(self, in_features=64):
        super(MainModel, self).__init__()
        self.up_proj = nn.Linear(in_features=in_features, out_features=2 * in_features)
        self.act = nn.ReLU()
        self.down_proj = nn.Linear(in_features=2 * in_features, out_features=in_features)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


def set_dist_env():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        import socket
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("localhost", 0))
            port = s.getsockname()[1]

        os.environ["MASTER_PORT"] = str(port)
        os.environ["HCCL_DETERMINISTIC"] = "1"

    print(f'Setting master: {os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}')


def dist_setup(rank, world_size, mesh_shape=None, mesh_dim_names=None):
    set_seed(12)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if mesh_shape is None:
        mesh_shape = (world_size,)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    device_mesh = init_device_mesh("npu", mesh_shape, mesh_dim_names=mesh_dim_names)
    torch.npu.set_device(rank)
    return device_mesh


def cleanup():
    dist.destroy_process_group()


def get_dp_dataloader(rank, world_size):
    dataset = RandomDataset(size=64, length=1024)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
