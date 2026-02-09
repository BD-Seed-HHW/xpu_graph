import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage


class PassManager:
    def __init__(self, config, **kwargs):
        self._config = config
        self._stage = FxStage.inference
        self._passes = []
        self._enable_passes = []

        from .optimizer import Optimizer

        Optimizer._debug = self._config.debug
        Optimizer._dump_graph = self._config.dump_graph

        from .patterns.pattern_manager import PatternManager

        self._pattern_manager = PatternManager(self._config)

        if self._config.bucketing_and_reordering:
            from .bucketing_and_reordering import BucketingAndReordering
            from .reshard_after_forward import ReshardAfterForward

            if ReshardAfterForward._opt_level <= self._config.opt_level:
                self._passes.append(ReshardAfterForward())

            if BucketingAndReordering._opt_level <= self._config.opt_level:
                assert "module_bucket_plans" in kwargs, "module_bucket_plans must be provided when bucketing_and_reordering is enabled"
                module_bucket_plans = kwargs.get("module_bucket_plans")
                self._passes.append(BucketingAndReordering(module_bucket_plans=module_bucket_plans))

        from .remove_runtime_assertions import RemoveAssertions

        if RemoveAssertions._opt_level <= self._config.opt_level:
            self._passes.append(RemoveAssertions())

        from .view_to_reshape import ViewToReshape

        if ViewToReshape._opt_level <= self._config.opt_level:
            self._passes.append(ViewToReshape())

        from .dce import Dce

        if Dce._opt_level <= self._config.opt_level:
            self._passes.append(Dce())

        from .cse import Cse

        # FIXME(zhangjihang): CSE will introduce some accurancy problem during pregrad stage, so I just skip it for safety now.
        if Cse._opt_level <= self._config.opt_level:
            self._passes.append(Cse())

        self._passes.append(self._pattern_manager)
        if self._config.constant_folding:
            from .constant_folding import ConstantFolding

            if ConstantFolding._opt_level <= self._config.opt_level:
                self._passes.append(ConstantFolding(self._config.folding_freezed_params))

        for pass_ in self._passes:
            pass_._set_level(self._config.opt_level)

    def reset_enable_passes_with_stage(self, stage: FxStage):
        self._enable_passes = []
        for pass_ in self._passes:
            enable_pass = pass_.get_pass_with_stage(stage)
            if enable_pass:
                self._enable_passes.append(enable_pass)

    def __call__(self, gm: fx.GraphModule, example_inputs, stage: FxStage):
        # Set pattern_manager to run stage-specific passes
        self.reset_enable_passes_with_stage(stage)

        changed = True
        while changed:
            from torch._guards import detect_fake_mode
            from torch._subclasses.fake_tensor import FakeTensor

            from xpu_graph.passes.fake_tensor_prop import FakeTensorProp

            assert all([isinstance(inp, FakeTensor) for inp in example_inputs if isinstance(inp, torch.Tensor)])
            fake_mode = detect_fake_mode(example_inputs)

            FakeTensorProp(gm, fake_mode).propagate_dont_convert_inputs(*example_inputs)

            changed = False
            for optimizer in self._enable_passes:
                changed = changed or optimizer(gm)

        return gm

    def get_pattern_manager(self):
        return self._pattern_manager
