import torch
import xpu_graph.aligner
from test_aligner_models import P6DenseConfig, P6DenseModel
from xpu_graph.aligner import AlignedModelGenerator

from copy import deepcopy


class RandTokenizer:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __call__(self, prompts: list[str], return_tensors: str = "pt", padding: bool = True):
        inputs = {
            "input_ids": torch.randint(0, self.vocab_size, (len(prompts), self.max_seq_len)),
            "attention_mask": torch.ones((len(prompts), self.max_seq_len), dtype=torch.long),
        }
        if return_tensors == "pt":
            return {k: v.to(device) for k, v in inputs.items()}
        else:
            raise ValueError(f"return_tensors={return_tensors} is not supported.")

def _get_device():
    try:
        import importlib
        importlib.import_module("torch_npu")
        return torch.device("npu:0")
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

if __name__ == "__main__":
    torch.manual_seed(42)
    device = _get_device()

    config = P6DenseConfig(
        vocab_size=64000,
        hidden_size=2048,
        intermediate_size=4096,
        num_hidden_layers=1,
        num_attention_heads=8,
        hidden_act="silu",
        _attn_implementation = "sdpa"
    )

    with torch.random.fork_rng():
        model_1: P6DenseModel = P6DenseModel(config).to(device)

    with torch.random.fork_rng():
        model_2: P6DenseModel = P6DenseModel(config).to(device)

    prompts = ["小炒肉怎么做？", "今天天气怎么样？"]

    tokenizer: RandTokenizer = RandTokenizer(vocab_size=config.vocab_size, max_seq_len=128)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs.update({"flash_attn_kwargs": {}})

    torch._dynamo.mark_dynamic(inputs["input_ids"], 0)
    torch._dynamo.mark_dynamic(inputs["input_ids"], 1)
    torch._dynamo.mark_dynamic(inputs["attention_mask"], 0)
    torch._dynamo.mark_dynamic(inputs["attention_mask"], 1)

    model_1 = AlignedModelGenerator("P6Dense", "eager_32").get_eager(model_1)
    model_2 = AlignedModelGenerator("P6Dense", "comp_32").get_compiled(model_2, args=(), kwargs=deepcopy(inputs))

    generated_tokens_1 = model_1(**deepcopy(inputs))
    generated_tokens_2 = model_2(**deepcopy(inputs))

    def _do_backward(model_out):
        flat_out, _ = torch.utils._pytree.tree_flatten(model_out)
        tensors_req = [v for v in flat_out if isinstance(v, torch.Tensor) and v.requires_grad]
        for t in tensors_req:
            t.backward(torch.ones_like(t))

    _do_backward(generated_tokens_1)
    _do_backward(generated_tokens_2)

    xpu_graph.aligner.mgr.print_data("P6Dense", ["eager_32", "comp_32"], gold_vid="eager_32")
    xpu_graph.aligner.mgr.export_dot("P6Dense", ["eager_32", "comp_32"], gold_vid="eager_32", steps=[0], fpath="p6d_gen.dot")
    xpu_graph.aligner.mgr.export_viewer(
        "P6Dense",
        ["eager_32", "comp_32"],
        gold_vid="eager_32",
        steps=[0],
        out_dir="./viewer/p6d_gen",
    )