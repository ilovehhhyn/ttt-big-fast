"""Configuration dataclasses.

Every field is explicit and validated in __post_init__. There are no silent
defaults that change behaviour: an invalid combination raises immediately.
"""

from __future__ import annotations

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
    rope: RopeConfig = field(default_factory=RopeConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0, "GQA needs heads divisible by kv heads"
        assert 0 <= self.fast_blocks <= self.num_layers
        # k >= b: the window must cover a whole chunk so the model can see
        # within-chunk context before TTT updates the weights (paper 2.3).
        assert self.window_size >= self.chunk_size, "window_size must be >= chunk_size"

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
    """

    optimizer: Literal["none", "normalized_sgd", "adamw", "muon", "clipped_sgd"] = "normalized_sgd"
    lr_rms: float = 1e-3
    norm_scope: Literal["tensor", "global"] = "tensor"
    eps_norm: float = 1e-6  # normalized_sgd denominator floor
    eps: float = 1e-8  # adamw epsilon
    beta1: float = 0.9
    beta2: float = 0.9
    warm_start: bool = True
    learned_lr: bool = True  # per-tensor log-multiplier, a slow parameter
    delta_decay: float = 0.0  # lambda: W <- W0 + (1-lambda)(W - W0) before each step
    lr_warmup_frac: float = 0.1  # fraction of outer steps to ramp lr_rms 0.1x -> 1x
    clip_tau: float = 1.0  # clipped_sgd only: global-norm clip threshold (e2e uses 1.0)

    def __post_init__(self) -> None:
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
        assert 0.0 <= self.warmup_frac < 1.0


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 8192
    tokens_per_step: int = 524288  # 0.5M tokens per outer step
    micro_batch: int = 1  # sequences per forward; fast weights are per-sequence
    remat_group: int = 0  # 0 -> round(sqrt(num_chunks))
    slow_spec: tuple[str, ...] = ("lora_A", "lora_B", "norm.weight", "inner_lr_log")
    seed: int = 0
    dtype: Literal["bf16", "fp32"] = "bf16"

    def __post_init__(self) -> None:
        assert self.micro_batch >= 1
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

    @property
    def num_chunks(self) -> int:
        return self.train.seq_len // self.model.chunk_size
