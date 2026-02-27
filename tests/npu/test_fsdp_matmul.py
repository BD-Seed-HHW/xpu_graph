import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from xpu_graph import OptLevel, Target, XpuGraph, XpuGraphConfig

from tests.npu.test_dist_utils import (
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
    model.npu(rank)
    model = FSDP(model, use_orig_params=True, device_mesh=device_mesh)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataloader = get_dp_dataloader(rank, world_size)

    if do_compile:
        xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=True,
                freeze=False,
                target=Target.npu,
                opt_level=OptLevel.level0,
                debug=True,
                vendor_compiler_config=None,
            )
        )
        model = torch.compile(model, backend=xpu_graph_backend, dynamic=False)

    for epoch in range(5):
        final_loss = 0
        n_batch = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.npu(rank), target.npu(rank)

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


def fsdp_test(ModCls, model_path="fsdp_model.pth"):
    set_dist_env()
    mp.set_start_method("spawn", force=True)
    world_size = torch.npu.device_count()
    return_queue1 = mp.Queue()
    return_queue2 = mp.Queue()
    model = ModCls()
    torch.save(model.state_dict(), model_path)

    do_compile = 0
    mp.spawn(
        train,
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
        train,
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
class TestFSDP:
    @pytest.fixture(autouse=True, scope="class")
    def setup_env(self, env_dispatch, request):
        pass

    @pytest.mark.parametrize(
        "PatternModel",
        [
            MainModel,
        ],
    )
    def test_fsdp(self, tmp_path, PatternModel):
        fsdp_test(PatternModel, tmp_path / "fsdp_model.pth")


if __name__ == "__main__":
    fsdp_test(MainModel)
