from .logging import get_logger


logger = get_logger(__name__)


from .modeling_flash_attention_utils import GPUFlashAttentionKwargs as FlashAttentionKwargs
from .modeling_flash_attention_utils import _flash_attention_forward
from .modeling_flash_attention_utils import get_gpu_flash_attn_kwargs as get_flash_attn_kwargs

# from .import_utils import (
#     is_fla_available,
#     is_fused_moe_available,
#     is_liger_kernel_available,
#     is_omnistore_available,
#     is_torch_version_greater_than_2_2,
#     is_torch_version_greater_than_2_4,
#     is_xperf_gpt_available,
# )
# from .modeling_fused_moe import fused_moe_forward
