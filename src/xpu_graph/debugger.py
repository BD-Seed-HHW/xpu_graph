from typing import Any

from torch.fx import Interpreter
from torch.fx._compatibility import compatibility
from torch.fx.node import Argument, Target


class Debugger(Interpreter):
    '''
    A debugger for tracing FX graph execution.

    Extends torch.fx.Interpreter to enable custom debugging logic during graph interpretation.
    '''

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        out = super().call_function(target, args, kwargs)
        return out
