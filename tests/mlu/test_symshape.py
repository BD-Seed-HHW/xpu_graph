import pytest
import torch
import torch._dynamo.config as dynamo_config
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn_view(query):
    """
    This is a mimic process of unpadding + target_attention + padding
    Assure capture_dynamic_output_shape_ops=True for putting dynamic-shape in FxGraph
    """

    bsz, qnum, dim = query.shape

    padding_mask = torch.ones([bsz, qnum], device=query.device, dtype=torch.int32)

    # 1. the unpadding part, reshape the padding mask 1D as the select mask, concat all unpadded sequences allong the seq dim
    flat_mask = torch.reshape(padding_mask, [-1])
    unpad_indices = torch.nonzero(flat_mask).squeeze(1)
    unpad_query = torch.reshape(query, [-1, dim])[unpad_indices]
    # 2. (skip the nested_concat\targeet_attention\nested_split part).
    # 3. the repadding part, repad the query outs
    #   By human knowledge we known the query mask is the same as the num of query tokens, but
    #   dynamo cannot infer this out directly. Thus, this reshape performs as a shape assertion
    query_out = unpad_query.reshape(bsz, qnum, dim)

    return query_out.view(-1, qnum * dim)


@dynamo_config.patch(capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True)
def fn_test(xpu_graph_backend, func):
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)
    for bsz in [8, 80, 100]:
        query = torch.randn(bsz, 2, 32, device=device, dtype=data_type)
        res = func(query)
        res1 = compiled(query)
        assert is_similar(res.cpu().float(), res1.cpu().float())


class TestSymShape:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn_view,
        ],
    )
    def test_symshape_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs():
            fn_test(self.xpu_graph_backend, pattern_func)

        assert "torch.ops.aten.nonzero" in caplog.text
        assert caplog.text.count("Pattern.FoldView1 changed graph") == 2


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, debug=True)
    fn_test(xpu_graph_backend, fn_view)
