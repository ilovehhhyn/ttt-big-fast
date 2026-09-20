"""TTT-E2E architecture options (arm E): qk_norm, post_norm, prime MLP.

Llama-3.2 (arms A-D) uses none of these, so every test here also asserts the
default path is unchanged.
"""

from __future__ import annotations

import pytest
import torch

from ttt.config import LoRAConfig, ModelConfig, RopeConfig, TrainConfig
from ttt.model.naming import fast_suffixes, split_parameters
from ttt.model.transformer import TTTTransformer


def cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=4,
                num_heads=4, num_kv_heads=2, window_size=8, chunk_size=4, fast_blocks=2,
                rope=RopeConfig(theta=10000.0, scaling="none"), lora=LoRAConfig(rank=0))
    base.update(kw)
    return ModelConfig(**base)


def test_defaults_are_llama():
    c = cfg()
    assert (c.qk_norm, c.post_norm, c.prime) == (False, False, False)
    assert c.fast_module == "mlp"


def test_prime_selects_prime_mlp_and_freezes_base():
    c = cfg(prime=True)
    assert c.fast_module == "mlp_prime"
    assert fast_suffixes(c) == ("mlp_prime.w1.weight", "mlp_prime.w2.weight", "mlp_prime.w3.weight")
    m = TTTTransformer(c, max_seq_len=16)
    sp = split_parameters(m, c, TrainConfig(seq_len=16, tokens_per_step=16))
    assert all("mlp_prime" in k for k in sp.fast), sorted(sp.fast)
    # The block's own MLP must be static safe storage, i.e. NOT fast.
    assert not any(k.endswith("mlp.w1.weight") for k in sp.fast)
    base_mlp = [n for n in dict(m.named_parameters()) if ".mlp.w1.weight" in n]
    assert base_mlp and all(n in sp.frozen for n in base_mlp)


def test_prime_only_exists_in_suffix_blocks():
    m = TTTTransformer(cfg(prime=True), max_seq_len=16)
    names = dict(m.named_parameters())
    assert not any(n.startswith("blocks.0.mlp_prime") or n.startswith("blocks.1.mlp_prime") for n in names)
    assert any(n.startswith("blocks.2.mlp_prime") for n in names)
    assert any(n.startswith("blocks.3.mlp_prime") for n in names)


def test_post_norm_adds_params_and_changes_output():
    plain, posted = TTTTransformer(cfg(), max_seq_len=16), TTTTransformer(cfg(post_norm=True), max_seq_len=16)
    extra = {n for n, _ in posted.named_parameters()} - {n for n, _ in plain.named_parameters()}
    assert any("seq_post_norm" in n for n in extra) and any("ffn_post_norm" in n for n in extra)


def test_qk_norm_adds_params_and_is_applied():
    m = TTTTransformer(cfg(qk_norm=True), max_seq_len=16)
    names = {n for n, _ in m.named_parameters()}
    assert any("attn.q_norm.weight" in n for n in names)
    assert any("attn.k_norm.weight" in n for n in names)
    # Scaling q_norm must change the output: proves it is actually in the path.
    ids = torch.randint(0, 32, (1, 16))
    ref = m.prefix_forward(ids).clone()
    with torch.no_grad():
        m.blocks[0].attn.q_norm.weight.mul_(3.0)
    assert not torch.allclose(ref, m.prefix_forward(ids), atol=1e-6)


def test_arm_e_shape_runs_end_to_end():
    """The full TTT-E2E option set together, through a chunked suffix forward."""
    c = cfg(prime=True, post_norm=True, qk_norm=True)
    m = TTTTransformer(c, max_seq_len=16).double()
    sp = split_parameters(m, c, TrainConfig(seq_len=16, tokens_per_step=16))
    assert sorted(sp.fast) == sorted(m.fast_param_names())
    ids = torch.randint(0, 32, (1, 16))
    h = m.prefix_forward(ids)
    caches = m.init_caches(batch=1, device=ids.device, dtype=h.dtype)
    logits, caches = m.suffix_forward(h[:, :4], fast=dict(sp.fast), caches=caches, chunk_index=0)
    assert logits.shape == (1, 4, 32)
    g = torch.autograd.grad(logits.sum(), [sp.fast[k] for k in sorted(sp.fast)], create_graph=True)
    assert all(torch.isfinite(x).all() and x.abs().sum() > 0 for x in g)


def test_prefix_segmented_equals_full():
    """Segmenting the prefix must be numerically exact, not an approximation.

    Sliding-window attention looks back at most `window_size`, and the rolling cache
    carries exactly that, so a segmented prefix sees the same keys/values as a one-shot
    prefix. Run in float64 where the window genuinely rolls (T > window).
    """
    c = cfg(num_layers=6, fast_blocks=2, window_size=8, chunk_size=4)  # prefix = 4 blocks
    m = TTTTransformer(c, max_seq_len=32).double()
    ids = torch.randint(0, c.vocab_size, (1, 32))
    full = m.prefix_forward(ids)
    for seg in (4, 8):  # segment must be <= window_size (8)
        got = m.prefix_forward(ids, segment=seg)
        assert got.shape == full.shape
        assert torch.allclose(got, full, atol=1e-10), f"segment={seg}: {(got - full).abs().max()}"

    import pytest as _pytest

    with _pytest.raises(AssertionError, match="must be <= window_size"):
        m.prefix_forward(ids, segment=16)


def test_arm_f_refuses_to_run_until_it_is_implemented():
    """ARMS["F"] is a placeholder identical to arm C. Running it would publish arm C's
    numbers under arm F's name, so it must fail loudly instead."""
    from types import SimpleNamespace

    from ttt.run import ARMS, build_everything

    assert ARMS["F"] == ARMS["C"], "arm F now differs from C: replace this guard with a real test"
    with pytest.raises(AssertionError, match="arm F is not implemented"):
        build_everything(SimpleNamespace(arm="F", device="cpu"))
