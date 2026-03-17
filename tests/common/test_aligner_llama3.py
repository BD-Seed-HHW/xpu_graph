import torch
import xpu_graph.aligner
from test_aligner_models.llama3 import Llama3
from xpu_graph.aligner import AlignedModelGenerator


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

    llama_config = {
        "vocab_size": 64,
        "dim": 128,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 4,
    }

    with torch.random.fork_rng():
        cpu_32 = AlignedModelGenerator("Llama3", "cpu_32").get_eager(Llama3(**llama_config))

    with torch.random.fork_rng():
        eager_32 = AlignedModelGenerator("Llama3", "eager_32").get_eager(Llama3(**llama_config).to(device))

    with torch.random.fork_rng():
        compile_32 = AlignedModelGenerator("Llama3", "compile_32").get_compiled(Llama3(**llama_config).to(device), (torch.randint(0, 64, (4, 16)).to(device),), {})

    with torch.random.fork_rng():
        eager_16 = AlignedModelGenerator("Llama3", "eager_16").get_eager(Llama3(**llama_config).half().to(device))

    with torch.random.fork_rng():
        compile_16 = AlignedModelGenerator("Llama3", "compile_16").get_compiled(Llama3(**llama_config).half().to(device), (torch.randint(0, 64, (4, 16)).to(device),), {})


    for i in range(2):
        x = torch.randint(0, 64, (4, 16))

        y = cpu_32(x.clone())
        y.backward(torch.ones_like(y))

        y = eager_32(x.clone().to(device))
        y.backward(torch.ones_like(y))

        y = compile_32(x.clone().to(device))
        y.backward(torch.ones_like(y))

        y = eager_16(x.clone().to(device))
        y.backward(torch.ones_like(y))

        y = compile_16(x.clone().to(device))
        y.backward(torch.ones_like(y))

        for model in [cpu_32, eager_32, compile_32, eager_16, compile_16]:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.add_(p.grad, alpha=-0.0001)
                        p.grad = None

    xpu_graph.aligner.mgr.print_data("Llama3", ["cpu_32", "eager_32", "compile_32", "eager_16", "compile_16"], gold_vid="cpu_32")
    xpu_graph.aligner.mgr.export_dot("Llama3", ["cpu_32", "eager_32", "compile_32", "eager_16", "compile_16"], gold_vid="cpu_32", steps=[0,1], fpath="llama3_align.dot")
    
    xpu_graph.aligner.mgr.export_viewer(
        "Llama3",
        ["cpu_32", "eager_32", "compile_32", "eager_16", "compile_16"],
        gold_vid="cpu_32",
        steps=[0, 1],
        out_dir="./viewer/llama3",
    )