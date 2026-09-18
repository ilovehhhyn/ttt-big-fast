"""SwiGLU feed-forward network (the tensor the TTT inner loop updates)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ttt.config import LoRAConfig

__all__ = ["SwiGLUMLP", "make_projection"]


def make_projection(name: str, in_features: int, out_features: int, lora: LoRAConfig | None) -> nn.Module:
    """A bias-free linear projection, upgraded to `LoRALinear` when `name` is a LoRA target.

    Shared by `SwiGLUMLP` and `SlidingWindowAttention` so both obey `cfg.lora.targets`
    with identical semantics. The `LoRALinear` import is deferred so that a rank-0
    configuration never depends on `ttt/model/lora.py`.
    """
    if lora is None or lora.rank == 0 or name not in lora.targets:
        return nn.Linear(in_features, out_features, bias=False)
    from ttt.model.lora import LoRALinear

    return LoRALinear(in_features, out_features, lora, bias=False)


class SwiGLUMLP(nn.Module):
    """Llama feed-forward block.

        y = w2( silu(w1(x)) * w3(x) )

    Llama naming: w1 = gate_proj, w3 = up_proj, w2 = down_proj.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, lora: LoRAConfig | None = None) -> None:
        super().__init__()
        assert hidden_size > 0 and intermediate_size > 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w1 = make_projection("w1", hidden_size, intermediate_size, lora)
        self.w3 = make_projection("w3", hidden_size, intermediate_size, lora)
        self.w2 = make_projection("w2", intermediate_size, hidden_size, lora)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.hidden_size, (
            f"SwiGLUMLP expects last dim {self.hidden_size}, got {x.shape[-1]}"
        )
        # SwiGLU: y = w2(silu(w1(x)) * w3(x)),  silu(z) = z * sigmoid(z)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
