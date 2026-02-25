import math

import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

DEVICE = "mlu"
dtype = torch.half


class _sdpa_pattern_tensor_scale:
    @staticmethod
    def fn(query, key, value, inv_scale):
        # query:bsz, self.num_heads, q_len, head_dim
        return torch.matmul(query, key.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(value)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        inv_scale = torch.Tensor([math.sqrt(Eqk)]).to(dtype=dtype, device=DEVICE)
        return (q, k, v, inv_scale)


class _sdpa_pattern_1:
    @staticmethod
    def fn(query, key, value, inv_scale):
        # query:bsz, self.num_heads, q_len, head_dim
        return torch.matmul(query, key.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(value)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        inv_scale = math.sqrt(Eqk)
        return (q, k, v, inv_scale)


class _sdpa_pattern_1_1:
    @staticmethod
    def fn(query, key, value, inv_scale):
        # query:bsz, self.num_heads, q_len, head_dim
        return (
            torch.matmul(query, key.transpose(-2, -1))
            .div(torch.clamp(torch.tensor([inv_scale], dtype=query.dtype, device=query.device), 0, 20))
            .softmax(dim=-1)
            .matmul(value)
        )

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        return _sdpa_pattern_1.gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev)


class _sdpa_pattern_2:
    @staticmethod
    def fn(query, key, value, scale_factor):
        return torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1).matmul(value)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        scale_factor = 1 / math.sqrt(Eqk)
        return (q, k, v, scale_factor)


class _sdpa_pattern_3:
    @staticmethod
    def fn(query, key, value, inv_scale_factor, dropout_p=0.0):
        return torch.nn.functional.dropout(
            torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
            p=dropout_p,
        ).matmul(value)

    gen = _sdpa_pattern_1.gen


class _sdpa_pattern_4:
    @staticmethod
    def fn(query, key, value, scale_factor, dropout_p=0.0):
        return torch.nn.functional.dropout(
            torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
            p=dropout_p,
        ).matmul(value)

    gen = _sdpa_pattern_2.gen


class _sdpa_pattern_5:
    @staticmethod
    def fn(query, key, value, attn_mask):
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
        # attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ value

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        attn_mask = torch.randn(B, Hq, Sq, Skv, dtype=dtype, device=DEVICE)
        return (q, k, v, attn_mask)


class _sdpa_pattern_5_1:
    @staticmethod
    def fn(query, key, value):
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))), dim=-1)
        # attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ value

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        return q, k, v


class _sdpa_pattern_5_2:
    fn = _sdpa_pattern_5.fn

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        # qkv:3d, mask:4d
        q = torch.randn(Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        attn_mask = torch.randn(B, Hq, Sq, Skv, dtype=dtype, device=DEVICE)
        return (q, k, v, attn_mask)


class _sdpa_pattern_6:
    @staticmethod
    def fn(query, key, value, attn_mask, dropout_p=0.0):
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        return attn_weight @ value

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        attn_mask = torch.randn(B, Hq, Sq, Skv, dtype=dtype, device=DEVICE)
        dropout_p = 0.0
        return (q, k, v, attn_mask, dropout_p)


class _sdpa_pattern_6_1:
    @staticmethod
    def fn(query, key, value, attn_mask, dropout_p=0.0):
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) - attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        return attn_weight @ value

    gen = _sdpa_pattern_6.gen


class _sdpa_pattern_7:
    @staticmethod
    def fn(query, key, value, dropout_p=0.0):
        # in real workloads inputs to matmul are permuted
        # causing matmul to expand to a series of expand and clone calls
        # we want the same to happen during pattern tracing
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        attn_weight = attn_weight.to(torch.float16)
        return attn_weight @ v

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Sq, Hq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Skv, Hkv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Skv, Hkv, Ev, dtype=dtype, device=DEVICE)
        return q, k, v


class _sdpa_pattern_8:
    @staticmethod
    def fn(query, key, value):
        # no dropout version of pattern 7
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)
        return attn_weight @ v

    gen = _sdpa_pattern_7.gen


class _sdpa_pattern_9:
    @staticmethod
    def fn(query, key, value, dropout_p=0.0):
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        q = q / math.sqrt(q.size(-1))
        div = q @ k.transpose(-2, -1)
        div = div.to(torch.float32)
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        attn_weight = attn_weight.to(torch.float16)
        return attn_weight @ v

    gen = _sdpa_pattern_7.gen


class _sdpa_pattern_10:
    @staticmethod
    def fn(query, key, value):
        # no dropout version of 9
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        q = q / math.sqrt(q.size(-1))
        div = q @ k.transpose(-2, -1)
        div = div.to(torch.float32)
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)
        return attn_weight @ v

    gen = _sdpa_pattern_7.gen


class _sdpa_pattern_11:
    @staticmethod
    def fn(query, key, value, inv_scale):
        # Mainly for huggingface models
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        return torch.matmul(q, k.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(v)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Sq, Hq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Skv, Hkv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Skv, Hkv, Ev, dtype=dtype, device=DEVICE)
        inv_scale = math.sqrt(Eqk)
        return (q, k, v, inv_scale)


class _sdpa_pattern_12:
    @staticmethod
    def fn(query, key, value, inv_scale_factor, dropout_p=0.0):
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        return torch.nn.functional.dropout(
            torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
            p=dropout_p,
        ).matmul(v)

    gen = _sdpa_pattern_11.gen


class _sdpa_pattern_13:
    @staticmethod
    def fn(query, key, value, dropout_p=0.0):
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
        return torch.bmm(attn_weight, value)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        return (q, k, v)


class _sdpa_pattern_14:
    @staticmethod
    def fn(query, key, value, inv_scale, attn_mask):
        # for BertLarge
        # Permutations are needed to create clones in graph.
        q = query.permute([0, 2, 1, 3])
        k = key.permute([0, 2, 1, 3])
        v = value.permute([0, 2, 1, 3])
        return (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1).matmul(v)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Sq, Hq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Skv, Hkv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Skv, Hkv, Ev, dtype=dtype, device=DEVICE)
        inv_scale = math.sqrt(Eqk)
        attn_bias = torch.randn((B, Hq, Sq, Skv), dtype=dtype, device=DEVICE)
        return (q, k, v, inv_scale, attn_bias)


# TODO
class _sdpa_pattern_15:
    @staticmethod
    def fn(query, key, value, inv_scale, attn_mask):
        # for DistilBert
        # Permutations are needed to create clones in graph.
        # Ref: https://github.com/pytorch/pytorch/issues/119911
        q = query.permute([0, 2, 1, 3])
        k = key.permute([0, 2, 1, 3])
        v = value.permute([0, 2, 1, 3])
        bs = q.size(0)
        k_len = k.size(-2)
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        return torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Sq, Hq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Skv, Hkv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Skv, Hkv, Ev, dtype=dtype, device=DEVICE)
        inv_scale = math.sqrt(Eqk)
        attn_bias = torch.randn((B, Hq, Sq, Skv), dtype=dtype, device=DEVICE)
        return (q, k, v, inv_scale, attn_bias)


class _sdpa_pattern_16:
    @staticmethod
    def fn(query, key, value, inv_scale, attn_mask, dropout_p=0.0):
        # for BertLarge with dropout
        q = query.permute([0, 2, 1, 3])
        k = key.permute([0, 2, 1, 3])
        v = value.permute([0, 2, 1, 3])
        return (
            torch.nn.functional.dropout(
                (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1),
                dropout_p,
            )
            .to(dtype=query.dtype)
            .matmul(v)
        )

    gen = _sdpa_pattern_14.gen


class _sdpa_pattern_16_1:
    @staticmethod
    def fn(query, key, value, inv_scale, attn_mask, dropout_p=0.0):
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        return (
            torch.nn.functional.dropout(
                (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1),
                dropout_p,
            )
            .to(dtype=query.dtype)
            .matmul(v)
        )

    gen = _sdpa_pattern_14.gen


class _sdpa_pattern_16_2:
    @staticmethod
    def fn(query, key, value, inv_scale, attn_mask, dropout_p=0.0):
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        return (
            torch.nn.functional.dropout(
                (torch.matmul(q.div(inv_scale), k.transpose(-2, -1)) + attn_mask).softmax(dim=-1),
                dropout_p,
            )
            .to(dtype=query.dtype)
            .matmul(v)
        )

    gen = _sdpa_pattern_14.gen


class _sdpa_pattern_16_3:
    @staticmethod
    def fn(query, key, value, inv_scale, attn_mask, dropout_p=0.0):
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        return (
            torch.nn.functional.dropout(
                (torch.matmul(q.div(inv_scale), k.transpose(-2, -1)) + attn_mask).softmax(dim=-1),
                dropout_p,
            )
            .to(dtype=query.dtype)
            .matmul(v)
        ).transpose(1, 2)

    gen = _sdpa_pattern_14.gen


class _sdpa_pattern_17:
    @staticmethod
    def fn(query, key, value, attn_mask, inv_scale, dropout_p):
        # for DistilBert with dropout
        q = query.permute([0, 2, 1, 3])
        k = key.permute([0, 2, 1, 3])
        v = value.permute([0, 2, 1, 3])
        bs = q.size(0)
        k_len = k.size(-2)
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        return (
            torch.nn.functional.dropout(torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p) @ v
        )

    gen = _sdpa_pattern_15.gen


class _sdpa_pattern_18:
    @staticmethod
    def fn(query, key, value, causal_mask, dropout_p=0.0):
        # for hf_GPT2 with dropout (introduces clone node) for inference
        # it also returns permuted key & value
        query = query.permute([0, 2, 1, 3])
        key = key.permute([0, 2, 1, 3])
        value = value.permute([0, 2, 1, 3])
        attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
        inv_scale = torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )
        attn_weights = attn_weights.div(inv_scale)
        causal_mask_value = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
        attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
        return (
            (torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(value)),
            key,
            value,
        )

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Sq, Hq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Skv, Hkv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Skv, Hkv, Ev, dtype=dtype, device=DEVICE)
        causal_mask = torch.tril(torch.ones((Sq, Skv), dtype=torch.bool, device=DEVICE)).view(1, 1, Sq, Skv)
        return (q, k, v, causal_mask)


class _sdpa_pattern_19:
    @staticmethod
    def fn(query, key, value, causal_mask, attn_mask, dropout_p=0.0):
        # for token-classification+gpt2 / text-generation+gpt2
        attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
        inv_scale = torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )
        attn_weights = attn_weights.div(inv_scale)
        causal_mask_value = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
        attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
        attn_weights = attn_weights + attn_mask
        attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
        return torch.nn.functional.dropout(attn_weights, dropout_p).matmul(value)

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Sq, Hq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Skv, Hkv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Skv, Hkv, Ev, dtype=dtype, device=DEVICE)
        causal_mask = torch.tril(torch.ones((Sq, Skv), dtype=torch.bool, device=DEVICE)).view(1, 1, Sq, Skv)
        attn_mask = torch.randn(B, 1, Sq, Skv, dtype=dtype, device=DEVICE)
        return (q, k, v, causal_mask, attn_mask)


class _sdpa_pattern_transformer_1:
    @staticmethod
    def fn(query, key, value):
        # llama
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=False)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        return (q, k, v)


class _sdpa_pattern_transformer_2:
    @staticmethod
    def fn(query, key, value, attention_mask):
        # llama/qwen/mixtral
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=False)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        attention_mask = torch.randn((B, Hq, Sq, Skv), dtype=dtype, device=DEVICE)
        return (q, k, v, attention_mask)


class _sdpa_pattern_transformer_3:
    @staticmethod
    def fn(query, key, value, attention_mask):
        # falcon
        attention_scores = query @ key.transpose(-1, -2)
        attention_scores /= math.sqrt(query.size(-1))

        attention_scores = torch.nn.functional.softmax(attention_scores + attention_mask, dim=-1, dtype=query.dtype)
        # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
        attn_output = attention_scores @ value
        return attn_output

    @staticmethod
    def gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev):
        q = torch.randn(B, Hq, Sq, Eqk, dtype=dtype, device=DEVICE)
        k = torch.randn(B, Hkv, Skv, Eqk, dtype=dtype, device=DEVICE)
        v = torch.randn(B, Hkv, Skv, Ev, dtype=dtype, device=DEVICE)
        attention_mask = torch.randn((B, Hq, Sq, Skv), dtype=dtype, device=DEVICE)
        return (q, k, v, attention_mask)


def fa_test(xpu_graph_backend, pattern):
    func = pattern.fn
    Eqk, Ev = 64, 64
    Sq, Skv = 38, 38
    Hq, Hkv = 32, 32

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)
    for B in [1, 2, 4]:
        args = pattern.gen(dtype, DEVICE, B, Sq, Skv, Hq, Hkv, Eqk, Ev)
        res = compiled(*args)
        res1 = func(*args)

        is_similar(res.cpu().float(), res1.cpu().float())


class TestFA:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            freeze=False,
            constant_folding=True,
            folding_freezed_params=False,
            opt_level=OptLevel.level2,
        )

    @pytest.mark.parametrize(
        "pattern",
        [
            _sdpa_pattern_tensor_scale,
            _sdpa_pattern_transformer_1,
            _sdpa_pattern_transformer_2,
            _sdpa_pattern_transformer_3,
            _sdpa_pattern_1,
            _sdpa_pattern_1_1,
            _sdpa_pattern_2,
            _sdpa_pattern_3,
            _sdpa_pattern_4,
            _sdpa_pattern_5,
            _sdpa_pattern_5_1,
            _sdpa_pattern_6,
            _sdpa_pattern_6_1,
            _sdpa_pattern_7,
            _sdpa_pattern_8,
            _sdpa_pattern_9,
            _sdpa_pattern_10,
            _sdpa_pattern_11,
            _sdpa_pattern_12,
            _sdpa_pattern_13,
        ],
    )
    def test_sdpa_patterns(self, caplog, pattern):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            fa_test(self.xpu_graph_backend, pattern)
        assert "Pattern.FusedSDPA changed graph" in caplog.text
        if pattern in [_sdpa_pattern_tensor_scale]:
            assert "Unwrap scale " in caplog.text


class TestFAWithTAOScale:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            freeze=False,
            constant_folding=True,
            folding_freezed_params=False,
            opt_level=OptLevel.level3,
        )

    @pytest.mark.parametrize(
        "pattern",
        [
            _sdpa_pattern_tensor_scale,
        ],
    )
    def test_sdpa_patterns(self, caplog, pattern):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            fa_test(self.xpu_graph_backend, pattern)
        assert "Pattern.FusedSDPA changed graph" in caplog.text
        assert "Unwrap scale " not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
        freeze=False,
        constant_folding=True,
        folding_freezed_params=False,
        opt_level=OptLevel.level2,
        debug=True,
    )
    fa_test(xpu_graph_backend, _sdpa_pattern_tensor_scale)
    fa_test(xpu_graph_backend, _sdpa_pattern_1)
    fa_test(xpu_graph_backend, _sdpa_pattern_1_1)
    fa_test(xpu_graph_backend, _sdpa_pattern_2)
    fa_test(xpu_graph_backend, _sdpa_pattern_3)
    fa_test(xpu_graph_backend, _sdpa_pattern_4)
    fa_test(xpu_graph_backend, _sdpa_pattern_5)
    fa_test(xpu_graph_backend, _sdpa_pattern_6)
    fa_test(xpu_graph_backend, _sdpa_pattern_7)
    fa_test(xpu_graph_backend, _sdpa_pattern_8)
    fa_test(xpu_graph_backend, _sdpa_pattern_9)
    fa_test(xpu_graph_backend, _sdpa_pattern_10)
    fa_test(xpu_graph_backend, _sdpa_pattern_11)
    fa_test(xpu_graph_backend, _sdpa_pattern_12)
    fa_test(xpu_graph_backend, _sdpa_pattern_13)
    fa_test(xpu_graph_backend, _sdpa_pattern_5_1)
    fa_test(xpu_graph_backend, _sdpa_pattern_transformer_1)
    fa_test(xpu_graph_backend, _sdpa_pattern_transformer_2)
    fa_test(xpu_graph_backend, _sdpa_pattern_transformer_3)
    fa_test(xpu_graph_backend, _sdpa_pattern_6_1)
