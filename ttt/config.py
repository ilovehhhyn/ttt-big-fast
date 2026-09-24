"""Configuration dataclasses.

Every field is explicit and validated in __post_init__. There are no silent
defaults that change behaviour: an invalid combination raises immediately.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RopeConfig:
    """Llama-3 scaled RoPE.

    Inverse frequencies f_j = 1 / theta^(2j/d) are rescaled by wavelength
    lambda_j = 2*pi/f_j:
        lambda_j > orig/low_freq_factor            -> f_j / factor      (low freq)
        lambda_j < orig/high_freq_factor           -> f_j               (high freq)
        otherwise                                  -> smooth interpolation
    """

    theta: float = 500000.0
    scaling: Literal["none", "llama3"] = "llama3"
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position: int = 8192
    # False = Llama/HF halves convention; True = TTT-E2E adjacent-pair (complex) form.
    interleaved: bool = False

    def __post_init__(self) -> None:
        if self.scaling == "llama3":
            assert self.factor > 1.0, "llama3 scaling needs factor > 1"
            assert self.high_freq_factor > self.low_freq_factor > 0.0


@dataclass(frozen=True)
class LoRAConfig:
    """Low-rank slow adapters.

    delta_W = scale * B @ A, with B initialised to zero so the model starts
    exactly at the pretrained point. scale = alpha/sqrt(r) (rsLoRA) or alpha/r.
    """

    rank: int = 0  # 0 disables LoRA entirely
    alpha: float = 16.0
    scaling: Literal["rslora", "classic"] = "rslora"
    targets: tuple[str, ...] = ("wq", "wk", "wv", "wo")

    def __post_init__(self) -> None:
        assert self.rank >= 0
        if self.rank > 0:
            assert len(self.targets) > 0, "lora.rank > 0 but no targets"
            allowed = {"wq", "wk", "wv", "wo", "w1", "w2", "w3"}
            unknown = set(self.targets) - allowed
            assert not unknown, f"unknown lora targets: {sorted(unknown)}"

    @property
    def scale(self) -> float:
        if self.rank == 0:
            return 0.0
        return self.alpha / (self.rank**0.5 if self.scaling == "rslora" else self.rank)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_layers: int = 16
    num_heads: int = 32
    num_kv_heads: int = 8
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    window_size: int = 8192
    chunk_size: int = 1024
    fast_blocks: int = 4  # number of trailing blocks whose MLPs are fast weights
    # --- TTT-E2E (arm E) architecture options. Llama-3.2 uses none of these. ---
    qk_norm: bool = False  # RMSNorm on q and k per head before RoPE
    post_norm: bool = False  # extra RMSNorm on each sublayer output (pre+post norm)
    prime: bool = False  # see `fast_module`
    # Arm F only (require prime). prime_intermediate_size: hidden width of the prime MLP
    # (None: the block's own intermediate_size, as arm E's checkpoint has it). prime_gate:
    # prime_out = gate * RMSNorm(mlp_prime(x)) with a scalar gate per block that starts at 0,
    # so the pretrained block is untouched at step 0 (LaCT, arXiv 2505.23884, Alg. 2, App. C.3).
    prime_intermediate_size: int | None = None
    prime_gate: bool = False
    # Initial value of every prime gate. 0.0 is LaCT's choice; a nonzero value is a deliberate
    # deviation for short runs, where a zero gate gives the prime MLP no gradient to start from.
    prime_gate_init: float = 0.0
    # Per-token learning rates on the fast-weight write (ttt/model/token_rate.py); a slow
    # parameter, so the run's slow_spec must include "token_rate" (ttt.run enforces it).
    token_rates: bool = False
    rope: RopeConfig = field(default_factory=RopeConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0, "GQA needs heads divisible by kv heads"
        assert 0 <= self.fast_blocks <= self.num_layers
        # k >= b: the window must cover a whole chunk so the model can see
        # within-chunk context before TTT updates the weights (paper 2.3).
        assert self.window_size >= self.chunk_size, "window_size must be >= chunk_size"
        assert self.prime_intermediate_size is None or self.prime, (
            f"prime_intermediate_size={self.prime_intermediate_size} requires prime=True; set prime or drop it"
        )
        assert self.prime_intermediate_size is None or self.prime_intermediate_size > 0
        assert not self.prime_gate or self.prime, "prime_gate=True requires prime=True; set prime or drop it"
        assert self.prime_gate_init == 0.0 or self.prime_gate, (
            f"prime_gate_init={self.prime_gate_init} requires prime_gate=True; set prime_gate or drop it"
        )

    @property
    def fast_module(self) -> str:
        """Which MLP the inner loop updates.

        prime=False (arms A-D): the block's own MLP is the fast weight.
        prime=True  (arm E/F):  a SECOND "prime" MLP is inserted in each suffix block
                                and is the fast weight, while the original MLP stays
                                static as "safe storage" for pretrained knowledge
                                (TTT-E2E 2.3.1, `feed_forward_prime` in their code).
        """
        return "mlp_prime" if self.prime else "mlp"

    @property
    def prime_intermediate(self) -> int:
        """Hidden width of the prime MLP: its own size, or the block's when none is set."""
        return self.intermediate_size if self.prime_intermediate_size is None else self.prime_intermediate_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def first_fast_layer(self) -> int:
        """Index of the first block with fast weights; blocks below are the frozen prefix."""
        return self.num_layers - self.fast_blocks


@dataclass(frozen=True)
class InnerConfig:
    """Inner-loop (test-time) optimizer.

    normalized_sgd:  W <- W - lr_rms * sqrt(numel) * g / (||g||_F + eps_norm)
                     so the per-element RMS of the update equals lr_rms.
    adamw:           differentiated-through AdamW, moments warm-started from the
                     first chunk gradient; denominator sqrt(v_hat + eps^2).
    muon:            W <- W - lr_rms * sqrt(max(m,n)) * NewtonSchulz5(g)
                     ns_dtype is the dtype of the five Newton-Schulz rounds only: float32
                     (default) or bfloat16 (Keller Jordan's reference Muon). The gradient,
                     the fast weights and the update stay in their own dtype.
    preconditioned_sgd:  normalized_sgd applied to D = g - (1 - shared_keep) * (g E) E^T, where the
                     columns of E [in, r] are the input ("key") directions that all tokens share
                     (scripts/key_basis.py). shared_keep = 1 is normalized_sgd exactly.
    """

    optimizer: Literal["none", "normalized_sgd", "adamw", "muon", "clipped_sgd", "preconditioned_sgd"] = "normalized_sgd"
    lr_rms: float = 1e-3
    norm_scope: Literal["tensor", "global"] = "tensor"
    eps_norm: float = 1e-6  # normalized_sgd denominator floor
    eps: float = 1e-8  # adamw epsilon
    beta1: float = 0.9
    beta2: float = 0.9
    warm_start: bool = True
    learned_lr: bool = True  # per-tensor log-multiplier, a slow parameter
    delta_decay: float = 0.0  # lambda: W <- W0 + (1-lambda)(W - W0) before each step
    lr_warmup_frac: float = 0.1  # fraction of outer steps to ramp lr_rms ilr_init -> 1x
    # e2e's ilr_init. DELIBERATE DEVIATION: their 760m/32K extension config sets
    # ilr_init: 1 (no inner-LR warmup at all). We start at 0.1 because our W_0 is a
    # pretrained Llama that was never trained to receive fast-weight updates, so the
    # first outer steps need the fast weights to move gently while the LoRA factors are
    # still near zero. Kept configurable so the deviation is visible and testable.
    ilr_init: float = 0.1
    clip_tau: float = 1.0  # clipped_sgd only: global-norm clip threshold (e2e uses 1.0)
    ns_dtype: Literal["float32", "bfloat16"] = "float32"  # muon only: dtype of the Newton-Schulz rounds
    # Applied by every inner rule after its update (LaCT, arXiv 2505.23884, Alg. 1 and 3).
    # row_reset: each row of a 2-D fast weight (one output unit's input vector) is rescaled to
    # the row norm it had before the step, so the update turns the row without changing its
    # length; 1-D tensors are left alone. none: the update is used as is.
    weight_norm: Literal["none", "row_reset"] = "none"
    # preconditioned_sgd only. A chunk gradient is G = sum_t d_t k_t^T (d_t: error at the matrix
    # output, k_t: its input, the "key"). Most of ||G||^2 lies in a few key directions shared by
    # all tokens; that part moves the output for every later token and caps the step size, while
    # the token-specific part is what stores "this context -> this next token".
    key_basis_path: str | None = None  # file written by scripts/key_basis.py
    shared_keep: float = 0.0  # fraction of the shared-direction component kept in the update

    def __post_init__(self) -> None:
        needs_basis = self.optimizer == "preconditioned_sgd"
        assert needs_basis == (self.key_basis_path is not None), (
            f"optimizer={self.optimizer!r} with key_basis_path={self.key_basis_path!r}: preconditioned_sgd "
            "requires a key basis and no other optimizer uses one; write it with scripts/key_basis.py and "
            "pass --key-basis, or drop the option"
        )
        assert 0.0 <= self.shared_keep <= 1.0, f"shared_keep must be in [0, 1], got {self.shared_keep}"
        assert self.ns_dtype == "float32" or self.optimizer == "muon", (
            f"ns_dtype={self.ns_dtype!r} with optimizer={self.optimizer!r}: only muon runs a Newton-Schulz "
            "iteration; pass --inner muon or drop --ns-dtype"
        )
        assert self.weight_norm in ("none", "row_reset"), (
            f"weight_norm={self.weight_norm!r}; must be 'none' or 'row_reset'"
        )
        assert self.lr_rms >= 0.0
        assert 0.0 <= self.delta_decay < 1.0
        assert self.eps > 0.0 and self.eps_norm > 0.0
        assert 0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0


@dataclass(frozen=True)
class OuterConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1  # applied to LoRA A/B only
    grad_clip: float = 1.0
    warmup_frac: float = 0.1
    end_lr: float = 1e-5
    total_steps: int = 2600

    def __post_init__(self) -> None:
        assert self.lr >= 0.0 and self.total_steps >= 1
        # Fail here, not inside the training loop: a warmup that rounds to zero is a
        # silent change of schedule (see resolve_warmup).
        resolve_warmup(self.warmup_frac, self.total_steps, "warmup_frac")


def smallest_steps_with_warmup(frac: float) -> int:
    """The smallest total_steps whose warmup round(frac * total_steps) is at least 1.

    Python rounds halves to even, so round(0.5) is 0: the first n with frac * n > 0.5 is
    the answer, not ceil(0.5 / frac) (5 for frac = 0.1, whose warmup rounds to 0).
    """
    assert 0.0 < frac < 1.0, f"frac must be in (0, 1), got {frac}"
    steps = math.ceil(0.5 / frac)
    while round(frac * steps) < 1:
        steps += 1
    return steps


def resolve_warmup(frac: float, total_steps: int, name: str) -> int:
    """Warmup length in STEPS, or a hard error if the configuration cannot deliver one.

        W = round(frac * total_steps)

    A configured warmup (frac > 0) that rounds to W = 0 is a BUG, not a no-op: it
    silently changes the optimisation schedule, so two runs that differ only in
    total_steps are no longer comparable. At total_steps=3 with frac=0.1 both the outer
    and the inner warmup vanished this way, and the run looked normal. Fail loudly
    instead; frac=0 remains the explicit way to ask for no warmup.
    """
    assert 0.0 <= frac < 1.0, f"{name} must be in [0, 1), got {frac}"
    assert total_steps >= 1, f"total_steps must be >= 1, got {total_steps}"
    if frac == 0.0:
        return 0
    warmup = round(frac * total_steps)
    assert warmup >= 1, (
        f"{name}={frac} over total_steps={total_steps} rounds to a 0-step warmup. "
        f"Use total_steps >= {smallest_steps_with_warmup(frac)}, or set {name}=0 to disable warmup "
        f"on purpose."
    )
    assert warmup < total_steps, (
        f"{name}={frac} gives warmup {warmup} >= total_steps {total_steps}"
    )
    return warmup


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 8192
    tokens_per_step: int = 524288  # 0.5M tokens per outer step
    micro_batch: int = 1  # sequences per forward; fast weights are per-sequence
    remat_group: int = 0  # 0 -> 1. Measured: larger groups use MORE memory (FINDINGS section 13)
    # Segment length for the frozen prefix. 0 = one shot over the whole sequence, which
    # is cheapest at short T; at 32K the one-shot prefix costs 72 GiB on its own, so it
    # must be segmented. Must divide seq_len and be <= window_size.
    prefix_segment: int = 0
    # Truncated backprop through time: detach the carry every `truncate_bptt` chunks so
    # the meta-gradient spans at most that many inner steps. 0 = no truncation (exact).
    # This makes peak memory O(truncate_bptt) instead of O(num_chunks), which is what
    # makes 32 chunks fit at all. The meta-gradient becomes BIASED: contributions from
    # inner steps further back than the window are dropped. PERK (arXiv:2507.06415) does
    # the same, unrolling only the last 1-2 of its 4 inner steps.
    truncate_bptt: int = 0
    slow_spec: tuple[str, ...] = ("lora_A", "lora_B", "norm.weight", "inner_lr_log")
    # Arm F: the fast weights' initial value W_0 is itself meta-learned. The outer optimizer
    # then owns the fast tensors as well (ParamSplit.outer); the inner loop still resets to
    # them at every sequence. False for arms A to D, whose W_0 is the pretrained MLP.
    fast_init_trained: bool = False
    seed: int = 0
    dtype: Literal["bf16", "fp32"] = "bf16"

    def __post_init__(self) -> None:
        assert self.micro_batch >= 1
        # "**" is the full-slow wildcard (arm D), not a substring pattern; combined with
        # other patterns it would be meaningless.
        assert "**" not in self.slow_spec or self.slow_spec == ("**",), (
            f'"**" must be the only entry of slow_spec, got {self.slow_spec!r}'
        )
        assert self.tokens_per_step % self.seq_len == 0, (
            f"tokens_per_step {self.tokens_per_step} must be a multiple of seq_len {self.seq_len}"
        )

    @property
    def seqs_per_step(self) -> int:
        return self.tokens_per_step // self.seq_len


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    inner: InnerConfig = field(default_factory=InnerConfig)
    outer: OuterConfig = field(default_factory=OuterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        assert self.train.seq_len % self.model.chunk_size == 0, (
            f"seq_len {self.train.seq_len} must be divisible by chunk_size {self.model.chunk_size}"
        )
        # The inner-LR warmup is a fraction of the OUTER step count, so only Config can
        # check it. Same rule as warmup_frac: rounding to zero is an error, not a skip.
        resolve_warmup(self.inner.lr_warmup_frac, self.outer.total_steps, "lr_warmup_frac")

    @property
    def num_chunks(self) -> int:
        return self.train.seq_len // self.model.chunk_size
