"""Low-rank slow adapters on the attention (and optionally MLP) projections.

In this project the MLP weight matrices are the FAST weights, rewritten by the
inner loop once per chunk. The LoRA factors here are SLOW weights: the outer
loop meta-learns them by differentiating through the inner loop, so they must
start the run at exactly the pretrained point.

Formulas
--------
Base linear (nn.Linear layout, weight is [out, in]):

    y = x W^T + b

LoRA update, with A in R^{r x in} and B in R^{out x r}:

    Delta_W = scale * (B @ A)                     shape [out, in]
    y       = x W^T + scale * (x A^T) B^T + b     == x (W + Delta_W)^T + b

The factored form is what `forward` computes: it costs r*(in + out) multiplies
per token instead of in*out, and it never materialises Delta_W.

Scaling
-------
    rsLoRA:   scale = alpha / sqrt(r)     (default)
    classic:  scale = alpha / r

rsLoRA (Kalajdzievski, arXiv:2312.03732, Thm 3.2) shows that the classic
alpha/r factor makes the gradient with respect to the adapter collapse like
1/r as the rank grows, so large-rank adapters learn slower than small-rank
ones at the same nominal learning rate. Dividing by sqrt(r) instead keeps the
forward activations and the adapter gradients rank-stable, which is what lets
us sweep the rank without retuning the outer learning rate.

Initialisation
--------------
    A ~ U(-1/sqrt(in_features), +1/sqrt(in_features))
    B  = 0

B = 0 makes Delta_W exactly zero at initialisation, so a freshly wrapped layer
is bitwise identical to the nn.Linear it wraps. A is non-zero because dL/dB
is proportional to A: a zero A would leave B with zero gradient forever.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ttt.config import LoRAConfig

__all__ = ["LoRALinear"]


class LoRALinear(nn.Module):
    """A linear layer with an additive low-rank adapter.

    Attributes
    ----------
    weight : [out_features, in_features] -- same layout and semantics as
        ``nn.Linear.weight``. Frozen base weight for attention projections; the
        FAST weight when this wraps an MLP projection.
    bias : [out_features] or None.
    lora_A : [rank, in_features] -- slow, uniform-initialised.
    lora_B : [out_features, rank] -- slow, zero-initialised.
    scale : float -- alpha/sqrt(r) (rsLoRA) or alpha/r (classic).
    cfg : the LoRAConfig this layer was built from.

    ``LoRAConfig(rank=0)`` is rejected. A rank-0 adapter is not a silent no-op
    here: it would allocate empty factors, add a constant zero to the output,
    and hide a configuration mistake behind a layer that merely looks adapted.
    Callers that want no adapter must use ``nn.Linear`` instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: LoRAConfig,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if cfg.rank <= 0:
            raise ValueError(
                f"LoRALinear requires cfg.rank > 0, got rank={cfg.rank}. "
                "A rank-0 adapter is a configuration error; use nn.Linear instead."
            )
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"bad shape: in_features={in_features}, out_features={out_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg
        self.scale: float = cfg.scale  # alpha/sqrt(r) or alpha/r; see module docstring

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.lora_A = nn.Parameter(torch.empty(cfg.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, cfg.rank))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Base weight: nn.Linear's own default, so a fresh LoRALinear built from
        # scratch (rather than from_linear) is distributed like the layer it replaces.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # A ~ U(+-1/sqrt(in_features)), B = 0  ->  Delta_W == 0 at init.
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.lora_A, -bound, bound)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        # y = x W^T + scale * ((x A^T) B^T) + b
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        low_rank = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        )
        return out + self.scale * low_rank

    def merged_weight(self) -> Tensor:
        """W + scale * (B @ A), shape [out_features, in_features].

        Used for deployment accounting (how many parameters actually ship) and
        for the SVD effective-rank diagnostic on the merged update.
        """
        return self.weight + self.scale * (self.lora_B @ self.lora_A)

    @classmethod
    def from_linear(cls, linear: nn.Linear, cfg: LoRAConfig) -> "LoRALinear":
        """Wrap an existing nn.Linear, copying its weight (and bias).

        The copy is storage-independent (``detach().clone()``): mutating the
        source linear afterwards must not touch the wrapped layer, and the
        wrapped layer's gradients must not flow back into it.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"from_linear expects nn.Linear, got {type(linear).__name__}")
        out_features, in_features = linear.weight.shape
        layer = cls(in_features, out_features, cfg, bias=linear.bias is not None)
        with torch.no_grad():
            layer.weight.copy_(linear.weight.detach().clone())
            if linear.bias is not None:
                assert layer.bias is not None
                layer.bias.copy_(linear.bias.detach().clone())
        return layer

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.cfg.rank}, "
            f"alpha={self.cfg.alpha}, scaling={self.cfg.scaling}, scale={self.scale:.6g}"
        )
