"""Per-token learning rates for the fast-weight write (LaCT, arXiv 2505.23884, Eq. 4).

A chunk's gradient for a fast matrix is G = sum_t d_t k_t^T: one outer product per token,
with k_t the matrix input and d_t the error at its output. `TokenRate` predicts a positive
weight eta_t = softplus(w . x_t + b) from the block input x_t, and `scale_gradient` applies
it so that the inner gradient becomes sum_t eta_t d_t k_t^T while the forward value is
unchanged. Under a normalized or Muon rule only the ratios between the eta_t matter (LaCT
Sec. 3.2). The rate is a slow parameter: the outer loop learns which tokens to write.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["TOKEN_RATE_INIT", "TOKEN_RATE_OFFSET", "TokenRate", "scale_gradient"]

#: eta at init for every token, so a fresh model equals one without token rates exactly.
TOKEN_RATE_INIT = 1.0
#: softplus(TOKEN_RATE_OFFSET) = TOKEN_RATE_INIT. Added in the forward as a Python float, so
#: it is rounded in the model's working dtype and not in the dtype the module was built in.
TOKEN_RATE_OFFSET = math.log(math.expm1(TOKEN_RATE_INIT))


def scale_gradient(y: Tensor, eta: Tensor) -> Tensor:
    """y in value; eta * dL/dy in the backward.  y [B, T, d], eta [B, T, 1] or [T, 1].

    The forward is y + (eta - 1) * (y - y.detach()), whose value is y exactly and whose
    gradient is dL/dy + (eta - 1) dL/dy = eta dL/dy. The multiply by eta is recorded when
    the inner gradient is taken with create_graph=True, so the meta-gradient reaches eta
    through the write and not through the value (which does not depend on eta).
    """
    assert eta.shape[-1] == 1 and eta.shape[:-1] == y.shape[: eta.ndim - 1], (
        f"eta must be [..., T, 1] for y {tuple(y.shape)}, got {tuple(eta.shape)}"
    )
    return y + (eta - 1.0) * (y - y.detach())


class TokenRate(nn.Module):
    """eta_t = softplus(w . x_t + b + TOKEN_RATE_OFFSET); w = 0 and b = 0 at init, so eta_t = 1."""

    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        assert hidden_size > 0
        self.linear = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        # The rate is computed in the parameters' own dtype (fp32 masters in every run),
        # never under autocast: it multiplies the gradient, and bf16 would quantise it to
        # steps of 2^-8 around 1.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.softplus(self.linear(x.to(self.linear.weight.dtype)) + TOKEN_RATE_OFFSET)
