from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from xpu_graph.utils import logger, setup_logger

from tests.npu.test_dist_utils import set_seed


@dataclass
class Qwen3MoeToyConfig:
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
    model_type: str = "qwen3_moe"
    num_attention_heads: int = 16
    num_hidden_layers: int = 2
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: None = None
    rope_theta: int = 1000000
    sliding_window: None = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936

    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1024
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list = None
    router_aux_loss_coef: float = 0.001

    is_training: bool = True

    def __post_init__(self):
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig, layer_idx: int):
        super().__init__()
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
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
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
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def init_weights(self, init_std):
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        for linear in (self.up_proj, self.down_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class Qwen3MoeExperts(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def init_weights(self, init_std):
        nn.init.trunc_normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    def init_weights(self, init_std):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig):
        super().__init__()
        self.config = config
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def init_weights(self, init_std):
        self.experts.init_weights(init_std)
        self.gate.init_weights(init_std)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.weight_init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        is_moe_layer = (
            (layer_idx not in config.mlp_only_layers) and
            (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)
        )
        if is_moe_layer:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def init_weights(self):
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(self.weight_init_std)
        self.mlp.init_weights(self.weight_init_std)


class Qwen3MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3MoeToyConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = config.head_dim

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


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    def create_mask(self, seq_length, device):
        causal_mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        return causal_mask

    def forward(self, input_ids: torch.LongTensor | None = None):
        if input_ids is None:
            raise ValueError("You must specify input_ids")

        input_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_embeds.shape[:2]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self.create_mask(seq_length, input_ids.device)

        for decoder_layer in self.layers:
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
        for layer in self.layers:
            layer.init_weights()


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen3MoeToyConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor | None = None):
        hidden_states = self.model(input_ids=input_ids)
        logits = self.lm_head(hidden_states)
        return F.softmax(logits, dim=-1)

    def init_weights(self):
        self.model.init_weights()
        self.lm_head.weight = self.model.embed_tokens.weight


if __name__ == "__main__":
    setup_logger(is_debug=True)
    set_seed(111)
    model_config = Qwen3MoeToyConfig()
    logger.info("Qwen3MoeToyConfig: %s\n", model_config)
    model = Qwen3MoeForCausalLM(model_config)
    model.init_weights()
    logger.info("Qwen3MoeForCausalLM: %s\n", model)
    input_ids = torch.randint(0, model_config.vocab_size, (1, 1024))
    out = model(input_ids)
    logger.info("Qwen3MoeForCausalLM output shape: %s\n", out.shape)
