import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE).
    RoPE encodes absolute positional information with a rotation matrix, which
    naturally incorporates explicit relative position dependency in self-attention.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq_const", self.inv_freq)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Returns:
            A tuple of (cos, sin) tensors for RoPE application.
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq_const)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq_const)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Return cosine and sine components
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def apply_rotary_embeddings(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embeddings to query and key tensors.
    Args:
        q, k: Query and Key tensors of shape (batch, seq_len, n_heads, head_dim)
        cos, sin: Precomputed cosine and sine tensors from RoPE.
    Returns:
        The rotated query and key tensors.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) mechanism. GQA is a generalization of
    Multi-Head and Multi-Query attention. It divides query heads into groups
    that share a single key and value head.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape for attention
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 3. Apply Rotary Position Embeddings
        xq, xk = apply_rotary_embeddings(xq, xk, cos, sin)

        # 4. Repeat K and V heads to match Q heads for GQA
        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)

        # 5. Transpose for attention calculation
        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. Scaled Dot-Product Attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        # 7. Concatenate heads and project out
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeats the key and value heads n_rep times to match the number of query heads.
        (bs, seq_len, n_kv_heads, head_dim) -> (bs, seq_len, n_heads, head_dim)
        """
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation function.
    It provides a gating mechanism that controls the information flow,
    often leading to better performance than standard ReLU.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """
    A single block of the Llama 3 transformer architecture.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = SwiGLU(dim, ffn_hidden_dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # Pre-normalization and self-attention
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        # Pre-normalization and feed-forward network
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama3(nn.Module):
    """
    A simplified Llama 3 model.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim // n_heads, max_seq_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, n_kv_heads, dim * 2)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        x = self.tok_embeddings(tokens)
        cos, sin = self.rope(x)

        # Causal mask to prevent attending to future tokens
        mask = self.create_causal_mask(tokens.size(1), x.device)

        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        x = self.norm(x)
        return self.output(x)

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    

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

    with torch.random.fork_rng():
        eager_32 = AlignedModelGenerator("Llama3", "eager_32").get_eager(Llama3(**llama_config))

    with torch.random.fork_rng():
        comp_32 = AlignedModelGenerator("Llama3", "comp_32").get_compiled(Llama3(**llama_config), (torch.randint(0, 64, (4, 16)),))

    with torch.random.fork_rng():
        eager_16 = AlignedModelGenerator("Llama3", "eager_16").get_eager(Llama3(**llama_config).half())
    
    with torch.random.fork_rng():
        comp_16 = AlignedModelGenerator("Llama3", "comp_16").get_compiled(Llama3(**llama_config).half(), (torch.randint(0, 64, (4, 16)),))


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