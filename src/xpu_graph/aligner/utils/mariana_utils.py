import torch
from mariana.trainer.callback.base import Callback
from ..alignment_manager import AlignmentManager
from typing import Any, Dict, List, Optional        
import os

import logging
logger = logging.getLogger("xpu_graph")


class XpuGraphAlignerMarianaCallback(Callback):
    def __init__(self, model_id: str, variant_id: str):
        self._xpuspeed_path = os.environ.get('XPUSPEED_PATH')
        self._mariana_path = self._xpuspeed_path + "/external/mariana"
        self.model_id = model_id
        self.variant_id = variant_id
        self.mgr = AlignmentManager()

    def on_step_start(self, trainer: Any, crs_module: "mariana.module.MarianaModule", global_step: int) -> None:
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            logger.info(f"[ALIGNER] on_step_start: {global_step}")

    def on_step_end(self, trainer: Any, crs_module: "mariana.module.MarianaModule", global_step: int) -> None:

        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            logger.info(f"[ALIGNER] on_step_end: {global_step}")
            for gid in [k for k in self.mgr.graphs.keys() if k.startswith(self.model_id)]:
                self.mgr.print_data(gid, [self.variant_id], gold_vid=self.variant_id)
                self.mgr.export_dot(gid, [self.variant_id], gold_vid=self.variant_id, steps=[0,1], fpath=f"{self._xpuspeed_path}/logs/aligner/{gid}_step{global_step}.dot")

        for gid in [k for k in self.mgr.graphs.keys() if k.startswith(self.model_id)]:
            self.mgr.get_graph(gid).clear_data()
