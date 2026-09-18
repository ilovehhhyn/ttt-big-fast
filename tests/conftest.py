"""Tiny CPU configs shared by every unit test."""

import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, RopeConfig, TrainConfig


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        window_size=8,
        chunk_size=4,
        fast_blocks=1,
        rope=RopeConfig(theta=10000.0, scaling="none"),
        lora=LoRAConfig(rank=2, alpha=4.0),
    )


@pytest.fixture
def tiny_cfg(tiny_model_cfg) -> Config:
    return Config(
        model=tiny_model_cfg,
        inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False),
        train=TrainConfig(seq_len=16, tokens_per_step=32, micro_batch=1, dtype="fp32"),
    )


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
