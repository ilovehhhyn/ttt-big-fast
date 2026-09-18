"""Weight-import tests.

The offline tests always run. The parity test needs the ~2.5GB checkpoint and is
marked `hf`; run it with `python -m pytest -m hf`.
"""

from __future__ import annotations

import pytest
import torch

from ttt.config import LoRAConfig
from ttt.utils.hf_import import MIRROR_REPO, hf_key_map, model_config_from_hf

LLAMA32_1B_CONFIG = {
    "model_type": "llama", "hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
    "vocab_size": 128256, "hidden_size": 2048, "intermediate_size": 8192,
    "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
    "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "rope_theta": 500000.0,
    "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                     "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
}


def test_config_from_hf_matches_plan():
    cfg = model_config_from_hf(LLAMA32_1B_CONFIG, window_size=8192, chunk_size=1024, fast_blocks=4)
    assert (cfg.num_layers, cfg.hidden_size, cfg.intermediate_size) == (16, 2048, 8192)
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (32, 8, 64)
    assert cfg.rope.scaling == "llama3" and cfg.rope.factor == 32.0
    assert cfg.first_fast_layer == 12  # last quarter of 16 blocks


def test_key_map_is_complete_and_injective():
    m = hf_key_map(16, tied=True)
    assert len(m) == 2 + 16 * 9, "2 globals + 9 tensors per block"
    assert len(set(m.values())) == len(m), "mapping must be injective"
    assert "lm_head.weight" not in m, "tied embeddings have no separate lm_head"
    assert hf_key_map(16, tied=False)["lm_head.weight"] == "lm_head.weight"


def test_key_map_covers_every_model_parameter():
    """Every non-LoRA parameter of our model must receive a pretrained tensor."""
    from ttt.model.transformer import TTTTransformer
    cfg = model_config_from_hf(LLAMA32_1B_CONFIG, window_size=8192, chunk_size=1024, fast_blocks=4)
    cfg_small = type(cfg)(**{**cfg.__dict__, "num_layers": 2, "vocab_size": 64,
                             "hidden_size": 32, "intermediate_size": 64,
                             "num_heads": 4, "num_kv_heads": 2, "window_size": 8, "chunk_size": 4,
                             "fast_blocks": 1})
    model = TTTTransformer(cfg_small, max_seq_len=16)
    targets = set(hf_key_map(cfg_small.num_layers, tied=True).values())
    own = {n for n, _ in model.named_parameters() if "lora_" not in n}
    assert own == targets, f"unmapped: {sorted(own - targets)} | extra: {sorted(targets - own)}"


@pytest.mark.hf
def test_logits_match_hf():
    """Our chunked prefix+suffix path must reproduce HF logits on a real checkpoint."""
    from transformers import AutoModelForCausalLM

    from ttt.model.naming import split_parameters
    from ttt.config import Config, InnerConfig, TrainConfig
    from ttt.optim.inner import build_inner_optimizer
    from ttt.train.inner_loop import TTTInnerLoop
    from ttt.utils.hf_import import build_llama_ttt

    seq_len, chunk = 256, 64
    # Exactly seq_len tokens: BOS then a repeating pattern, so every chunk is full.
    pattern = [791, 6864, 315, 9822, 374, 12366, 13]
    body = (pattern * (seq_len // len(pattern) + 1))[: seq_len - 1]
    ids = torch.tensor([[128000] + body], dtype=torch.long)
    assert ids.shape == (1, seq_len)

    hf = AutoModelForCausalLM.from_pretrained(MIRROR_REPO, torch_dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        ref = hf(ids).logits[0]

    model = build_llama_ttt(MIRROR_REPO, max_seq_len=seq_len, window_size=seq_len,
                            chunk_size=chunk, fast_blocks=4, dtype=torch.float32)
    model.eval()
    cfg = Config(model=model.cfg, inner=InnerConfig(optimizer="none"),
                 train=TrainConfig(seq_len=seq_len, tokens_per_step=seq_len, dtype="fp32"))
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    split = split_parameters(model, model.cfg, cfg.train)

    prefix = model.prefix_forward(ids)
    caches = model.init_caches(batch=1, device=ids.device, dtype=prefix.dtype)
    got = []
    for i in range(seq_len // chunk):
        sl = slice(i * chunk, (i + 1) * chunk)
        logits, caches = model.suffix_forward(prefix[:, sl], fast=dict(split.fast),
                                              caches=caches, chunk_index=i)
        got.append(logits[0])
    got = torch.cat(got, dim=0)
    assert got.shape == ref.shape
    err = (got - ref).abs().max().item()
    corr = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    print(f"max|delta|={err:.4f} corr={corr:.6f}")
    assert corr > 0.9999, f"logit correlation {corr}"
    assert err < 0.5, f"max abs logit error {err}"
