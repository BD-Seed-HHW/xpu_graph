import torch
from mariana.trainer.callback.base import Callback
from ..alignment_manager import AlignmentManager
from typing import Any, Dict, List, Optional        

import logging
logger = logging.getLogger("xpu_graph")


class XpuGraphAlignerMarianaCallback(Callback):
    def __init__(self, model_id: str, variant_id: str):
        self.model_id = model_id
        self.variant_id = variant_id
        self.mgr = AlignmentManager()

    def on_step_start(self, trainer: Any, crs_module: "mariana.module.MarianaModule", global_step: int) -> None:
        logger.info(f"[ALIGNER] on_step_start: {global_step}")

    def on_step_end(self, trainer: Any, crs_module: "mariana.module.MarianaModule", global_step: int) -> None:
        logger.info(f"[ALIGNER] on_step_end: {global_step}")

        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            self.mgr.print_data(self.model_id, [self.variant_id], gold_vid=self.variant_id)
        self.mgr.get_graph(self.model_id).clear_data()
