from torch import T
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from xpu_graph.utils import logger, setup_logger
from tests.npu.test_dist_utils import set_seed


# SimpleModelConfig from huggingface Qwen/Qwen3-0.6B/config.json
@dataclass
class Qwen3ToyConfig:
    """Configuration for SimpleModel."""
    # architectures: list = [
    #     "Qwen3ForCausalLM"
    # ]
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072  
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 2 # Number of hidden layers, original model is 28 layers
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: None = None
    rope_theta: int = 1000000
    sliding_window: None = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    # transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936

    # custom setting
    is_training: bool = True


# modeling_qwen3, copied and simplified from transformers

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # import pdb; pdb.set_trace()
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3ToyConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=True)  
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=True)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    # eager attention forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.config.is_training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
    
    def init_weights(self, init_std):
        for linear in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.o_proj.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3ToyConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx = layer_idx
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        # self.act_fn = nn.ReLU()

    def forward(self, x):
        out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return out

    def init_weights(self, init_std):
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        for linear in (self.up_proj, self.down_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3ToyConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config, layer_idx=layer_idx)
        self.weight_init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        # self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    def init_weights(self):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(self.weight_init_std)
        self.mlp.init_weights(self.weight_init_std)


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3ToyConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.head_dim = self.config.head_dim

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / self.head_dim)
        )
        self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.padding_idx = getattr(config, "pad_token_id", None)
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # delete padding_idx
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # self.has_sliding_layers = "sliding_attention" in self.config.layer_types

    def create_mask(self, seq_length, device):
        causal_mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
    ):
        if (input_ids is None):
            raise ValueError("You must specify exactly input_ids")

        input_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_embeds.shape[:2]
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self.create_mask(seq_length, input_ids.device)        
    
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    def init_weights(self):
        if self.embed_tokens is not None:
            nn.init.normal_(self.embed_tokens.weight)
        if self.norm is not None:
            self.norm.reset_parameters()


class Qwen3ForCausalLM(nn.Module):

    def __init__(self, config: Qwen3ToyConfig):
        super().__init__()
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
    ):
        hidden_states = self.model(input_ids=input_ids)
        logits = self.lm_head(hidden_states)
        return F.softmax(logits, dim=-1)

    def init_weights(self):
        self.model.init_weights()
        self.lm_head.weight = self.model.embed_tokens.weight


if __name__ == "__main__":
    setup_logger(is_debug=True)
    set_seed(111)
    model_config = Qwen3ToyConfig()
    logger.info("Qwen3ToyConfig: %s", model_config)
    model = Qwen3ForCausalLM(model_config)
    model.init_weights()
    input_ids = torch.randint(0, model_config.vocab_size, (1, 1024)) # shape: [B, S], range: [0, vocab_size)
    out = model(input_ids)
    logger.info("Qwen3ForCausalLM output shape: %s", out.shape)
    # print(model)
    # modules = list(model.modules())
    # for mod in modules:
    #     params_dict = dict(mod.named_parameters(recurse=False))
    #     import pdb; pdb.set_trace()
    #     for p_name, p in params_dict.items():
    #         print(f"param: {p_name}")
