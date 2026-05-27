# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
else:
    _flash_supports_window_size = None


def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    indices_k: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_in_batch_k: int,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa2_from_position_ids(position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cummulative lengths of each examples in the batch will be extracted from position_ids.

    NOTE: ideally cummulative lengths should be prepared at the data collator stage

    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        max_seqlen (`int`):
            The max sequence length for inputs when passing position ids.

    Return:
        query (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    _position_ids = position_ids.flatten()
    indices_q = torch.arange(_position_ids.size(0), device=_position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[_position_ids == 0],
            torch.tensor(_position_ids.size(), device=_position_ids.device, dtype=torch.int32),
        )
    )

    # Do not use position_ids.max() to compute since some VL models repeat the same position_id for image input.
    max_length = cu_seq_lens.diff().max().item()

    return (indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


class GPUFlashAttentionKwargs(TypedDict, total=False):
    indices_k: Optional[List[int]]

    max_seqlen_q: Optional[List[int]]
    max_seqlen_k: Optional[List[int]]
    cu_seqlens_q: Optional[List[int]]
    cu_seqlens_k: Optional[List[int]]

    is_position_ids_single_seq: Optional[bool]


def get_gpu_flash_attn_kwargs(
    cu_seqlens: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    position_ids: Optional[torch.Tensor] = None,
    query_length: int = None,
):
    flash_attn_kwargs = GPUFlashAttentionKwargs()
    if cu_seqlens is not None:
        max_seqlen_q = max_seqlen_k = cu_seqlens.diff().max().item()
        flash_attn_kwargs["max_seqlen_q"] = max_seqlen_q
        flash_attn_kwargs["max_seqlen_k"] = max_seqlen_k

    elif attention_mask is not None:
        indices_k, cu_seqlens_k, max_seqlen_k = _get_unpad_data(attention_mask=attention_mask)
        flash_attn_kwargs["indices_k"] = indices_k
        flash_attn_kwargs["cu_seqlens_k"] = cu_seqlens_k
        flash_attn_kwargs["max_seqlen_k"] = max_seqlen_k

    elif position_ids is not None:
        # Cache the result if it's just one sequence.
        is_single_seq = query_length == 1 or (torch.diff(position_ids, dim=-1) >= 0).all()
        # If it’s a tensor, run .item() to turn it into a Python bool to avoid device to host transfer when
        # checking the bollean value.
        if isinstance(is_single_seq, torch.Tensor):
            is_single_seq = is_single_seq.item()
        flash_attn_kwargs["is_position_ids_single_seq"] = is_single_seq

        # If only single sequence, no need to calculate cu_seqlens since we just fall back to BSH mode.
        if not flash_attn_kwargs["is_position_ids_single_seq"]:
            _, cu_seqlens, max_seq_lens = prepare_fa2_from_position_ids(position_ids=position_ids)
            flash_attn_kwargs["cu_seqlens_q"], flash_attn_kwargs["cu_seqlens_k"] = cu_seqlens
            flash_attn_kwargs["max_seqlen_q"], flash_attn_kwargs["max_seqlen_k"] = max_seq_lens
    else:
        flash_attn_kwargs = None

    return flash_attn_kwargs


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    flash_attn_kwargs: Optional[GPUFlashAttentionKwargs] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        max_seqlen (`int`):
            The max sequence length for inputs when passing position ids or cu_seqlens.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1

    # 2d mask is passed through the layers
    attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach.
    #
    # if it's already cached, no need to compute again since this is an expensive CPU blocking op.
    if position_ids is not None:
        is_position_ids_single_seq = (
            flash_attn_kwargs["is_position_ids_single_seq"]
            if flash_attn_kwargs is not None and "is_position_ids_single_seq" in flash_attn_kwargs
            else query_length == 1 or (torch.diff(position_ids, dim=-1) >= 0).all()
        )

    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens_k = cu_seqlens
        if flash_attn_kwargs is None:
            max_seqlen_q = max_seqlen_k = cu_seqlens.diff().max().item()
        else:
            max_seqlen_q = flash_attn_kwargs["max_seqlen_q"]
            max_seqlen_k = flash_attn_kwargs["max_seqlen_k"]

        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

    # Contains at least one padding token in the sequence
    elif attention_mask is not None:
        batch_size = query_states.shape[0]

        if flash_attn_kwargs is None:
            indices_k, cu_seqlens_k, max_seqlen_k = _get_unpad_data(attention_mask)
        else:
            indices_k = flash_attn_kwargs["indices_k"]
            cu_seqlens_k = flash_attn_kwargs["cu_seqlens_k"]
            max_seqlen_k = flash_attn_kwargs["max_seqlen_k"]

        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_layer=query_states,
            key_layer=key_states,
            value_layer=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            indices_k=indices_k,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_in_batch_k=max_seqlen_k,
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_q, max_seqlen_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    elif position_ids is not None and not is_position_ids_single_seq:
        batch_size = query_states.size(0)
        query_states = query_states.view(-1, query_states.size(-2), query_states.size(-1))
        key_states = key_states.view(-1, key_states.size(-2), key_states.size(-1))
        value_states = value_states.view(-1, value_states.size(-2), value_states.size(-1))

        if flash_attn_kwargs is None:
            _, cu_seqlens, max_seq_lens = prepare_fa2_from_position_ids(position_ids=position_ids)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
        else:
            cu_seqlens_q = flash_attn_kwargs["cu_seqlens_q"]
            cu_seqlens_k = flash_attn_kwargs["cu_seqlens_k"]
            max_seqlen_q = flash_attn_kwargs["max_seqlen_q"]
            max_seqlen_k = flash_attn_kwargs["max_seqlen_k"]

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return attn_output
