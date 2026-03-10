from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_p6d import P6DenseConfig
from .convert_p6d_checkpoint import (
    convert_p6_dense_to_xperf_compatible,
    convert_p6dense_config_from_cruise,
    p6_dense_convert_pt_to_hf,
    p6_dense_get_temporary_xperf_config,
)
from .modeling_p6d import (
    P6DenseForCausalLM,
    P6DenseForSequenceClassification,
    P6DenseForTokenClassification,
    P6DenseModel,
)


AutoConfig.register("seed_p6dense", P6DenseConfig)
AutoModel.register(P6DenseConfig, P6DenseModel)
AutoModelForCausalLM.register(P6DenseConfig, P6DenseForCausalLM)
AutoModelForSequenceClassification.register(P6DenseConfig, P6DenseForSequenceClassification)
AutoModelForTokenClassification.register(P6DenseConfig, P6DenseForTokenClassification)
