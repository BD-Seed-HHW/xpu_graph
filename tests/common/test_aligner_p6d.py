import torch
from transformers import AutoTokenizer, GPT2Tokenizer
from test_aligner_models import P6DenseConfig, P6DenseForCausalLM, P6DenseModel

import xpu_graph.aligner
from xpu_graph.aligner import AlignedModelGenerator

torch.manual_seed(42)

config = P6DenseConfig(
    vocab_size=64000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=1,
    num_attention_heads=32,
    num_key_value_heads=None,
    hidden_act="silu",
    _attn_implementation = "eager"
)

with torch.random.fork_rng(devices=[torch.get_default_device()]):
    model_1: P6DenseModel = P6DenseModel(config).cuda()

with torch.random.fork_rng(devices=[torch.get_default_device()]):
    model_2: P6DenseModel = P6DenseModel(config).cuda()

prompts = ["小炒肉怎么做？"]

tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
inputs.update({"flash_attn_kwargs": {}})

model_1 = AlignedModelGenerator("P6Dense", "eager_32").get_eager(model_1)
model_2 = AlignedModelGenerator("P6Dense", "comp_32").get_compiled(model_2, **{"args": (), "kwargs": inputs})

generated_tokens_1 = model_1(**inputs)
generated_tokens_2 = model_2(**inputs)

def _do_backward(model_out):
    flat_out, _ = torch.utils._pytree.tree_flatten(model_out)
    tensors_req = [v for v in flat_out if isinstance(v, torch.Tensor) and v.requires_grad]
    for t in tensors_req:
        t.backward(torch.ones_like(t))

_do_backward(generated_tokens_1)
_do_backward(generated_tokens_2)

xpu_graph.aligner.mgr.print_data("P6Dense", ["eager_32", "comp_32"], gold_vid="eager_32")
xpu_graph.aligner.mgr.export_dot("P6Dense", ["eager_32", "comp_32"], gold_vid="eager_32", steps=[0], fpath="p6d_align.dot")