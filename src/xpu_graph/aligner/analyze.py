from dataclasses import dataclass, field
from typing import Optional

import torch

from .alignment_graph import AlignmentGraph, AlignmentNode, GraphMapping, Stage
from .alignment_manager import AlignmentManager


class AlignmentAnalyzer:
    pass