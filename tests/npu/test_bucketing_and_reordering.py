import argparse
import os

import pytest
import torch
import torch.multiprocessing as mp
from xpu_graph.config import OptLevel, Target, XpuGraphConfig
from xpu_graph.utils import logger, setup_logger

from tests.npu.test_dist_utils import set_dist_env, set_seed
from tests.npu.train_utils import (
    ParallelizeDims,
    Qwen3ForCausalLM,
    Qwen3ToyConfig,
    TrainConfig,
    parallelize_model,
    train,
)

TORCH_DTYPE = torch.bfloat16
torch.set_default_dtype(TORCH_DTYPE)


TRAIN_CONFIG = TrainConfig(
    model_path="/tmp/test/weight.pt",
    dataset_path="/tmp/test/data.pt",
    parallelize_dims=None,
    model_config=Qwen3ToyConfig(),
    loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
    is_debug=True,
    device="npu",
    steps=200,
)

XPU_GRAPH_CONFIG = XpuGraphConfig(
    is_training=True,
    freeze=False,
    target=Target.npu,
    opt_level=OptLevel.level1,
    debug=True,
    bucketing_and_reordering=False,
    vendor_compiler_config=None,
)


def test_fsdp(path: str = "/tmp/test/fsdp.pt"):
    logger.info("begin test fsdp")
    set_seed(TRAIN_CONFIG.seed)
    set_dist_env()
    mp.set_start_method("spawn", force=True)
    train_config = TRAIN_CONFIG
    world_size_ = torch.npu.device_count()
    train_config.parallelize_dims = ParallelizeDims(
        dp_replicate=1,
        dp_shard=world_size_,
        cp=1,
        tp=1,
        pp=1,
        world_size=world_size_,
        device=train_config.device,
    )
    train_config.parallelize_fn = parallelize_model
    train_config.is_compile = True
    mp.spawn(
        train,
        args=(train_config, path),
        nprocs=world_size_)
    logger.info("fsdp training finished")


def test_fsdp_bucketing_and_reordering():
    logger.info("begin test fsdp bucketing and reordering")
    XPU_GRAPH_CONFIG.bucketing_and_reordering = True
    test_fsdp("/tmp/test/fsdp_bucketing_and_reordering.pt")
    XPU_GRAPH_CONFIG.bucketing_and_reordering = False


def test_no_fsdp(path: str = "/tmp/test/no_fsdp.pt"):
    logger.info("begin test no-fsdp")
    set_seed(TRAIN_CONFIG.seed)
    mp.set_start_method("spawn", force=True)
    train_config = TRAIN_CONFIG
    train_config.parallelize_dims = ParallelizeDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        world_size=1,
        device=train_config.device,
    )
    mp.spawn(
        train,
        args=(train_config, path),
        nprocs=1)
    logger.info("no-fsdp training finished")


def compare_weight(path1: str, path2: str, atol: float = 1e-4, rtol: float = 1e-4):
    if not os.path.exists(path1):
        logger.error(f"path1: {path1} not exists")
        return
    if not os.path.exists(path2):
        logger.error(f"path2: {path2} not exists")
        return
    logger.info(f"begin compare weight, path1: {path1}, path2: {path2}")
    state_dict1 = torch.load(path1, map_location="cpu")
    state_dict2 = torch.load(path2, map_location="cpu")
    assert state_dict1.keys() == state_dict2.keys(), "two state_dict keys are not equal"
    equal = True
    for k in state_dict1.keys():
        tensor1 = state_dict1[k]
        tensor2 = state_dict2[k]
        tensor1 = tensor1.detach().float().cpu()
        tensor2 = tensor2.detach().float().cpu()
        diff = (tensor1 - tensor2).abs()
        thr = atol + rtol * tensor2.abs()
        bad = diff > thr

        max_diff = diff.max().item()
        num_bad = bad.sum().item()
        if num_bad != 0:
            logger.error("max_diff: %s, num_bad: %s, weight %s is not close", max_diff, num_bad, k)
            equal = False
        else:
            logger.info("weight %s is close", k)
    if equal:
        logger.info("two state_dict weights are close")


def generate_data(folder: str = "/tmp/test"):
    vocab_size = TRAIN_CONFIG.model_config.vocab_size
    seq_len = TRAIN_CONFIG.max_seq_len
    if not os.path.exists(folder):
        os.makedirs(folder)
    tensor = torch.randint(0, vocab_size, size=(TRAIN_CONFIG.num_samples, seq_len))
    # tensor = torch.randint(0, vocab_size, size=(1, seq_len))
    torch.save(tensor, f"{folder}/data.pt")
    logger.info(f"save generate data to {folder}/data.pt")


def generate_weight_and_save(folder: str = "/tmp/test"):
    model = Qwen3ForCausalLM(Qwen3ToyConfig())
    model.init_weights()
    state_dict = model.state_dict()
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state_dict, f"{folder}/weight.pt")
    logger.info(f"save generate weight to {folder}/weight.pt")


def prepare_test_data():
    if os.path.exists("/tmp/test") and os.path.exists(TRAIN_CONFIG.model_path) and os.path.exists(TRAIN_CONFIG.dataset_path):
        logger.info("model weight and data already exists in folder /tmp/test")
    else:
        generate_data()
        generate_weight_and_save()


@pytest.mark.exclusive
def test_bucketing_and_reordering():
    setup_logger(is_debug=True)
    logger.info("begin test bucketing and reordering")
    prepare_test_data()
    test_fsdp()
    test_bucketing_and_reordering()
    compare_weight("/tmp/test/fsdp_bucketing_and_reordering.pt", "/tmp/test/no_fsdp.pt")


if __name__ == "__main__":
    setup_logger(is_debug=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default="all", choices=["fsdp", "no_fsdp", "all"])
    args = parser.parse_args()
    prepare_test_data()
    if args.test == "all":
        test_no_fsdp()
        test_fsdp()
        compare_weight("/tmp/test/no_fsdp.pt", "/tmp/test/fsdp.pt", atol=0.0, rtol=0.0)
    elif args.test == "fsdp":
        test_fsdp()
        test_fsdp_bucketing_and_reordering()
        compare_weight("/tmp/test/fsdp.pt", "/tmp/test/fsdp_bucketing_and_reordering.pt", atol=0.0, rtol=0.0)
    elif args.test == "no_fsdp":
        test_no_fsdp()
