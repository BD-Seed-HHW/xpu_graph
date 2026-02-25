import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import xpu_graph
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from xpu_graph.config import OptLevel

from tests.mlu.test_dist_utils import (
    MainModel,
    cleanup,
    dist_setup,
    get_dp_dataloader,
    set_dist_env,
)


def train(rank, world_size, do_compile, return_queue, ModCls, model_path):
    device_mesh = dist_setup(rank, world_size)
    model = ModCls()
    model.load_state_dict(torch.load(model_path))
    model.train()
    model.mlu(rank)
    model = parallelize_module(
        model,
        device_mesh,
        parallelize_plan={
            "up_proj": ColwiseParallel(input_layouts=Shard(0), output_layouts=Shard(1)),
            "down_proj": RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(0)),
        },
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataloader = get_dp_dataloader(rank, world_size)

    if do_compile:
        xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=True, freeze=False, opt_level=OptLevel.level2, debug=True
        )
        model = torch.compile(model, backend=xpu_graph_backend, dynamic=False, fullgraph=True)

    for epoch in range(5):
        final_loss = 0
        n_batch = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.mlu(rank), target.mlu(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

            final_loss = final_loss + loss.item()
            n_batch = n_batch + 1
        final_loss = final_loss / n_batch

    return_queue.put((rank, final_loss))
    cleanup()


def infer(rank, world_size, do_compile, return_queue, ModCls, model_path):
    device_mesh = dist_setup(rank, world_size)
    model = ModCls()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.mlu(rank)
    model = parallelize_module(
        model,
        device_mesh,
        parallelize_plan={
            "up_proj": ColwiseParallel(input_layouts=Shard(0), output_layouts=Shard(1)),
            "down_proj": RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(0)),
        },
    )
    criterion = nn.MSELoss(reduction="sum")
    dataloader = get_dp_dataloader(rank, world_size)

    if do_compile:
        xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=False, opt_level=OptLevel.level2, debug=True
        )
        model = torch.compile(model, backend=xpu_graph_backend, dynamic=False, fullgraph=True)

    final_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.mlu(rank), target.mlu(rank)
            output = model(data)
            loss = criterion(output, target)

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Batch [{batch_idx}], Loss: {loss.item():.4f}")

            final_loss = final_loss + loss

    return_queue.put((rank, final_loss.item()))
    cleanup()


def tp_test(ModCls, is_training=True, model_path="tp_model.pth"):
    set_dist_env()
    mp.set_start_method("spawn", force=True)
    world_size = torch.mlu.device_count()
    return_queue1 = mp.Queue()
    return_queue2 = mp.Queue()
    model = ModCls()
    torch.save(model.state_dict(), model_path)

    do_compile = 0
    mp.spawn(
        train if is_training else infer,
        args=(world_size, do_compile, return_queue1, ModCls, model_path),
        nprocs=world_size,
        join=True,
    )
    results1 = {}
    for _ in range(world_size):
        rank, loss = return_queue1.get()
        results1[rank] = loss

    do_compile = 1
    mp.spawn(
        train if is_training else infer,
        args=(world_size, do_compile, return_queue2, ModCls, model_path),
        nprocs=world_size,
        join=True,
    )
    results2 = {}
    for _ in range(world_size):
        rank, loss = return_queue2.get()
        results2[rank] = loss

    for i in range(world_size):
        assert abs(results1[i] - results2[i]) < 0.01


@pytest.mark.exclusive
@pytest.mark.env_settings([
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
])
class TestTP:
    @pytest.fixture(autouse=True, scope="class")
    def setup_env(self, env_dispatch, request):
        pass

    @pytest.mark.parametrize(
        "PatternModel",
        [
            MainModel,
        ],
    )
    @pytest.mark.parametrize(
        "is_training",
        [
            True,
            False,
        ],
    )
    def test_tp(self, tmp_path, PatternModel, is_training):
        tp_test(PatternModel, is_training, tmp_path / "tp_model.pth")


if __name__ == "__main__":
    tp_test(MainModel, is_training=True)
