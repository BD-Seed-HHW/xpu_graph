import os
import argparse
import pytest
import torch
import torch.multiprocessing as mp
from tests.npu.test_trainer.train import TrainConfig, train
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3ToyConfig
from parallel_dims import ParallelizeDims
from xpu_graph.utils import setup_logger, logger
from tests.npu.test_trainer.parallelize import parallelize_model
from tests.npu.test_dist_utils import set_dist_env, set_seed
from xpu_graph.config import Target, OptLevel, XpuGraphConfig

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
)
XPU_GRAPH_CONFIG = XpuGraphConfig(
    is_training=True,
    freeze=False,
    target=Target.npu,
    opt_level=OptLevel.level1,
    debug=True,
    overlap_manual_scheduling=True,
    vendor_compiler_config=None,
)


def test_fsdp():
    logger.info(f"begin test fsdp")
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
        args=(train_config,), 
        nprocs=world_size_)
    logger.info(f"fsdp training finished")


def test_no_fsdp():
    logger.info(f"begin test no-fsdp")
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
        args=(train_config,), 
        nprocs=1)
    logger.info(f"no-fsdp training finished")


def compare_weight(path1: str, path2: str):
    if not os.path.exists(path1):
        logger.error(f"path1: {path1} not exists")
        return
    if not os.path.exists(path2):
        logger.error(f"path2: {path2} not exists")
        return
    logger.info(f"begin compare weight, path1: {path1}, path2: {path2}")
    state_dict1 = torch.load(path1)
    state_dict2 = torch.load(path2)
    assert state_dict1.keys() == state_dict2.keys(), "two state_dict keys are not equal"
    equal = True
    for k in state_dict1.keys():
        tensor1 = state_dict1[k]
        tensor2 = state_dict2[k]
        if not torch.allclose(tensor1, tensor2, atol=1e-4, rtol=1e-4):
            max_diff = torch.max(torch.abs(tensor1 - tensor2))
            num_diff = torch.sum(torch.abs(tensor1 - tensor2) > 1e-4)
            logger.error("max_diff: %s, num_diff: %s, weight %s is not close, tensor1.dtype: %s, tensor2.dtype: %s\ntensor1: %s \ntensor2: %s", max_diff, num_diff, k, tensor1.dtype, tensor2.dtype, tensor1, tensor2)
            equal = False
            break
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


def test_all():
    test_no_fsdp()
    test_fsdp()
    compare_weight(f"/tmp/test/fsdp.pt", f"/tmp/test/no_fsdp.pt")


def prepare_test_data():
    if os.path.exists("/tmp/test") and os.path.exists(TRAIN_CONFIG.model_path) and os.path.exists(TRAIN_CONFIG.dataset_path):
        logger.info(f"model weight and data already exists in folder /tmp/test")
    else:
        generate_data()
        generate_weight_and_save()


@pytest.mark.exclusive
def test_bucketing_and_reordering():
    setup_logger(is_debug=True)
    logger.info(f"begin test bucketing and reordering")
    prepare_test_data()
    test_all()


if __name__ == "__main__":
    setup_logger(is_debug=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default="all", choices=["fsdp", "no_fsdp", "all"])
    args = parser.parse_args()
    prepare_test_data()
    if args.test == "all":
        test_all()
    elif args.test == "fsdp":
        test_fsdp()
    elif args.test == "no_fsdp":
        test_no_fsdp()
    compare_weight(f"/tmp/test/no_fsdp.pt", f"/tmp/test/fsdp.pt")
