"""Orbax/TensorStore import tests.

The offline tests build a SYNTHETIC pytree with the documented e2e key names and
stacked shapes, so they run without network, without orbax and in milliseconds.
The single real-checkpoint test is marked `hf` (it pulls ~4.9GB); run it with
`python -m pytest -m hf`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ttt.config import ModelConfig
from ttt.model.transformer import TTTTransformer
from ttt.utils.orbax_import import (
    E2E_REPO,
    STAGES,
    e2e_760m_config,
    map_e2e_to_ours,
)

# Small enough to build in milliseconds, with every dimension distinct so that a
# transposed or mis-stacked tensor cannot accidentally match.
SMALL = dict(
    num_layers=4, fast_blocks=2, hidden_size=8, intermediate_size=16,
    num_heads=2, num_kv_heads=2, vocab_size=32, window_size=8, chunk_size=4,
)

SRC = "language_model.model."
BLOCKS = SRC + "h.blocks."
PRIME = SRC + "h.prime_storage."


def synthetic_tree(cfg: ModelConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """The exact key set an orbax restore of an e2e checkpoint produces.

    Per-layer arrays carry a leading axis of `num_layers` because `Block` is built
    with `jax.vmap`; prime arrays are stacked over `suffix_len` instead. Linear
    weights are [in, out] (e2e's `NormalLinear` applies `x @ weight`).
    """
    rng = np.random.default_rng(seed)
    L, s = cfg.num_layers, cfg.fast_blocks
    d, f, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim

    def a(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape, dtype=np.float32)

    tree = {SRC + "wte.weight": a(cfg.vocab_size, d), SRC + "ln_f.weight": a(d)}
    for n in ("seq_norm", "ffn_norm", "seq_post_norm", "ffn_post_norm"):
        tree[BLOCKS + n + ".weight"] = a(L, d)
    for n in ("wq", "wk", "wv", "wo"):
        tree[BLOCKS + "seq_modeling_block." + n + ".weight"] = a(L, d, d)
    for n in ("q_norm", "k_norm"):
        tree[BLOCKS + "seq_modeling_block." + n + ".weight"] = a(L, hd)
    tree[BLOCKS + "feed_forward.w1.weight"] = a(L, d, f)
    tree[BLOCKS + "feed_forward.w2.weight"] = a(L, f, d)
    tree[BLOCKS + "feed_forward.w3.weight"] = a(L, d, f)
    for n in ("ffn_prime_norm", "ffn_prime_post_norm"):
        tree[PRIME + n + ".weight"] = a(s, d)
    tree[PRIME + "feed_forward_prime.w1.weight"] = a(s, d, f)
    tree[PRIME + "feed_forward_prime.w2.weight"] = a(s, f, d)
    tree[PRIME + "feed_forward_prime.w3.weight"] = a(s, d, f)
    return tree


def pretrained_param_names(model: TTTTransformer) -> set[str]:
    """Every parameter that must receive a checkpoint tensor.

    LoRA adapters and the learned inner-LR log scalars are new slow parameters with
    no e2e counterpart, so they are excluded by design (same rule as hf_import)."""
    return {
        n for n, _ in model.named_parameters()
        if "lora_" not in n and not n.startswith("inner_lr_log.")
    }


# --------------------------------------------------------------------------- config


def test_config_matches_resolved_yaml():
    cfg = e2e_760m_config()
    assert (cfg.num_layers, cfg.hidden_size, cfg.intermediate_size) == (24, 1536, 3328)
    assert (cfg.num_heads, cfg.num_kv_heads) == (16, 16), "e2e has no GQA"
    assert cfg.head_dim == 96
    assert cfg.vocab_size == 128256 and cfg.tie_word_embeddings is True
    assert cfg.rms_norm_eps == 1e-6
    assert (cfg.qk_norm, cfg.post_norm, cfg.prime) == (True, True, True)
    assert cfg.fast_blocks == 6 and cfg.first_fast_layer == 18
    assert cfg.rope.scaling == "none", "resolved_config has no rope scaling, NOT llama3"
    assert cfg.rope.theta == 500000.0
    assert (cfg.window_size, cfg.chunk_size) == (8192, 1024)
    assert cfg.lora.rank == 0, "LoRA is added later by the experiment, not by the import"


def test_overrides_are_applied_and_validated():
    cfg = e2e_760m_config(**SMALL)
    assert (cfg.num_layers, cfg.hidden_size, cfg.fast_blocks) == (4, 8, 2)
    assert cfg.qk_norm and cfg.post_norm and cfg.prime, "overrides must not drop arm E flags"
    with pytest.raises(AssertionError):
        e2e_760m_config(not_a_field=1)


def test_stage_table():
    assert E2E_REPO == "Luxel/ttt-e2e-760m-results"
    assert "s2_adapt" in STAGES
    assert STAGES["s2_adapt"].endswith("S2_ADAPT/adapt-760m-e2e-8K-from-fa")
    assert len(set(STAGES.values())) == len(STAGES) == 3


# ------------------------------------------------------------------- the load-bearing test


def test_unstack_and_transpose_roundtrip():
    """Every produced name/shape must match a freshly built TTTTransformer, both ways."""
    cfg = e2e_760m_config(**SMALL)
    model = TTTTransformer(cfg, max_seq_len=cfg.window_size)
    out = map_e2e_to_ours(synthetic_tree(cfg), cfg)

    own = dict(model.named_parameters())
    targets = pretrained_param_names(model)
    assert set(out) == targets, (
        f"unmapped: {sorted(targets - set(out))} | extra: {sorted(set(out) - targets)}"
    )
    for name, t in out.items():
        assert t.shape == own[name].shape, f"{name}: got {tuple(t.shape)}, want {tuple(own[name].shape)}"
        assert isinstance(t, torch.Tensor)

    # And it really loads.
    from ttt.utils.orbax_import import load_e2e_into_model
    load_e2e_into_model(model, synthetic_tree(cfg), cfg)


def test_transpose_is_actually_applied():
    """A non-square linear must come out [out, in] and hold the TRANSPOSED values."""
    cfg = e2e_760m_config(**SMALL)
    tree = synthetic_tree(cfg)
    out = map_e2e_to_ours(tree, cfg)

    src = tree[BLOCKS + "feed_forward.w1.weight"]  # [L, hidden, intermediate]
    assert src.shape == (cfg.num_layers, cfg.hidden_size, cfg.intermediate_size)
    got = out["blocks.1.mlp.w1.weight"]
    assert got.shape == (cfg.intermediate_size, cfg.hidden_size), "torch.nn.Linear is [out, in]"
    np.testing.assert_allclose(got.numpy(), src[1].T)
    # The untransposed slice would be a different (non-square) shape, so a silent
    # reshape would be caught above; check the values too in case in == out elsewhere.
    assert not np.allclose(got.numpy().T, src[1].T), "values must not equal the untransposed source"

    # wo is square (hidden -> hidden): shape alone cannot detect a missing transpose.
    src_wo = tree[BLOCKS + "seq_modeling_block.wo.weight"]
    np.testing.assert_allclose(out["blocks.2.attn.wo.weight"].numpy(), src_wo[2].T)
    assert not np.allclose(out["blocks.2.attn.wo.weight"].numpy(), src_wo[2])

    # Norms and the embedding are NOT transposed.
    np.testing.assert_allclose(out["blocks.3.seq_norm.weight"].numpy(),
                               tree[BLOCKS + "seq_norm.weight"][3])
    np.testing.assert_allclose(out["embed_tokens.weight"].numpy(), tree[SRC + "wte.weight"])


# ------------------------------------------------------------------------ strictness


@pytest.mark.parametrize("dropped", [
    SRC + "wte.weight",
    BLOCKS + "seq_modeling_block.q_norm.weight",
    PRIME + "feed_forward_prime.w2.weight",
])
def test_missing_key_raises(dropped):
    cfg = e2e_760m_config(**SMALL)
    tree = synthetic_tree(cfg)
    del tree[dropped]
    with pytest.raises(KeyError) as exc:
        map_e2e_to_ours(tree, cfg)
    msg = str(exc.value)
    assert dropped in msg, f"error must name the missing key, got: {msg}"
    # ...and list what IS there, so the mismatch is diagnosable.
    assert SRC + "ln_f.weight" in msg or BLOCKS + "ffn_norm.weight" in msg


def test_extra_key_raises():
    cfg = e2e_760m_config(**SMALL)
    tree = synthetic_tree(cfg)
    tree[SRC + "h.blocks.mystery.weight"] = np.zeros((cfg.num_layers, cfg.hidden_size), np.float32)
    with pytest.raises(KeyError) as exc:
        map_e2e_to_ours(tree, cfg)
    assert "mystery" in str(exc.value), "no tensor may be silently skipped"


def test_wrong_source_shape_raises():
    """A stack over the wrong axis length must not be unstacked silently."""
    cfg = e2e_760m_config(**SMALL)
    tree = synthetic_tree(cfg)
    tree[BLOCKS + "ffn_norm.weight"] = np.zeros((cfg.num_layers + 1, cfg.hidden_size), np.float32)
    with pytest.raises(AssertionError) as exc:
        map_e2e_to_ours(tree, cfg)
    assert "ffn_norm" in str(exc.value)


def test_load_rejects_a_model_built_from_a_different_config():
    from ttt.utils.orbax_import import load_e2e_into_model
    cfg = e2e_760m_config(**SMALL)
    other = e2e_760m_config(**{**SMALL, "intermediate_size": 24})
    model = TTTTransformer(other, max_seq_len=other.window_size)
    with pytest.raises(AssertionError) as exc:
        load_e2e_into_model(model, synthetic_tree(cfg), cfg)
    assert "w1" in str(exc.value) or "w2" in str(exc.value) or "w3" in str(exc.value)


# ----------------------------------------------------------------------------- prime


def test_prime_only_for_suffix_blocks():
    cfg = e2e_760m_config(**SMALL)
    out = map_e2e_to_ours(synthetic_tree(cfg), cfg)
    primed = {int(n.split(".")[1]) for n in out if ".mlp_prime." in n}
    assert primed == set(range(cfg.first_fast_layer, cfg.num_layers)) == {2, 3}
    for tag in ("ffn_prime_norm", "ffn_prime_post_norm"):
        idx = {int(n.split(".")[1]) for n in out if n.endswith(f".{tag}.weight")}
        assert idx == primed, f"{tag} landed on {sorted(idx)}, expected {sorted(primed)}"
    assert not any(n.startswith(("blocks.0.mlp_prime", "blocks.1.mlp_prime")) for n in out)


def test_prime_slice_order_follows_the_suffix():
    """prime_storage index j must land on block first_fast_layer + j, not block j."""
    cfg = e2e_760m_config(**SMALL)
    tree = synthetic_tree(cfg)
    out = map_e2e_to_ours(tree, cfg)
    src = tree[PRIME + "feed_forward_prime.w2.weight"]  # [suffix_len, intermediate, hidden]
    for j in range(cfg.fast_blocks):
        i = cfg.first_fast_layer + j
        np.testing.assert_allclose(out[f"blocks.{i}.mlp_prime.w2.weight"].numpy(), src[j].T)


# ------------------------------------------------------------------ real checkpoint (hf)


@pytest.mark.hf
def test_real_checkpoint_converts_with_full_coverage(tmp_path):
    """Download the real S2_ADAPT checkpoint, convert it, assert coverage both ways."""
    from ttt.utils.orbax_import import download_checkpoint, read_orbax_tree

    path = download_checkpoint(E2E_REPO, "s2_adapt")
    assert path.name == "model_weights" and path.is_dir(), path

    tree = read_orbax_tree(path)
    print("top-level keys:", sorted(tree))
    assert all(isinstance(v, np.ndarray) for v in tree.values())

    cfg = e2e_760m_config()
    out = map_e2e_to_ours(tree, cfg)
    model = TTTTransformer(cfg, max_seq_len=cfg.window_size)
    assert set(out) == pretrained_param_names(model)
    load_state = dict(model.named_parameters())
    for name, t in out.items():
        assert t.shape == load_state[name].shape, name
        assert torch.isfinite(t).all(), f"{name} has non-finite values"

    # Shapes that would silently differ if the vmap axis or the transpose were wrong.
    assert out["embed_tokens.weight"].shape == (128256, 1536)
    assert out["blocks.0.attn.wq.weight"].shape == (1536, 1536)
    assert out["blocks.23.mlp.w1.weight"].shape == (3328, 1536)
    assert out["blocks.18.mlp_prime.w2.weight"].shape == (1536, 3328)
    assert out["blocks.0.attn.q_norm.weight"].shape == (96,)
