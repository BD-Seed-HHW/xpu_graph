import torch
from test_aligner_models.llama3 import Llama3

import xpu_graph.aligner
from xpu_graph.aligner import AlignedModelGenerator


if __name__ == "__main__":
    torch.manual_seed(42)
    
    llama_config = {
        "vocab_size": 64,
        "dim": 128,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 4,
    }

    with torch.random.fork_rng(devices=[torch.get_default_device()]):
        eager_32 = AlignedModelGenerator("Llama3", "eager_32").get_eager(Llama3(**llama_config))

    with torch.random.fork_rng(devices=[torch.get_default_device()]):
        comp_32 = AlignedModelGenerator("Llama3", "comp_32").get_compiled(Llama3(**llama_config), (torch.randint(0, 64, (4, 16)),), {})

    with torch.random.fork_rng(devices=[torch.get_default_device()]):
        eager_16 = AlignedModelGenerator("Llama3", "eager_16").get_eager(Llama3(**llama_config).half())
    
    with torch.random.fork_rng(devices=[torch.get_default_device()]):
        comp_16 = AlignedModelGenerator("Llama3", "comp_16").get_compiled(Llama3(**llama_config).half(), (torch.randint(0, 64, (4, 16)),), {})


    for i in range(2):
        x = torch.randint(0, 64, (4, 16))

        y = eager_32(x.clone())
        y.backward(torch.ones_like(y))

        y = comp_32(x.clone())
        y.backward(torch.ones_like(y))

        y = eager_16(x.clone())
        y.backward(torch.ones_like(y))

        y = comp_16(x.clone())
        y.backward(torch.ones_like(y))

        # apply grad
        for model in [eager_32, comp_32, eager_16, comp_16]:
            for p in model.parameters():
                p.data.add_(p.grad*1e-4)
                if p.grad is not None:
                    p.grad.zero_()

    xpu_graph.aligner.mgr.print_data("Llama3", ["eager_32", "comp_32", "eager_16", "comp_16"], gold_vid="eager_32")
    xpu_graph.aligner.mgr.export_dot("Llama3", ["eager_32", "comp_32", "eager_16", "comp_16"], gold_vid="eager_32", steps=[0,1], fpath="llama3_align.dot")