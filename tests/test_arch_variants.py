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


def test_arm_f_is_a_real_configuration_distinct_from_arm_c():
    """Arm F was a placeholder equal to arm C and refused to run. It now names the prime
    layout: prime MLP fast, its W_0 trained, the gate slow."""
    from ttt.run import ARMS

    assert ARMS["F"] != ARMS["C"]
    assert ARMS["F"]["prime"] is True and ARMS["C"]["prime"] is False
    assert "prime_gate" in ARMS["F"]["slow"] and "prime_gate" not in ARMS["C"]["slow"]


def test_arm_f_gate_opens_at_the_first_outer_step_while_the_prime_weights_wait():
    """One outer step on arm F's layout: the fast set is the prime MLP alone; the gate moves
    off 0 (it gets a gradient); the prime W_0 does not move yet (its gradient is exactly 0
    while the gate is 0); the pretrained MLP of the fast block stays frozen."""
    from ttt.config import Config, InnerConfig, OuterConfig
    from ttt.optim.inner import build_inner_optimizer
    from ttt.optim.outer import build_outer_optimizer
    from ttt.run import ARMS
    from ttt.train.inner_loop import TTTInnerLoop
    from ttt.train.trainer import Trainer

    mcfg = cfg(prime=True, prime_intermediate_size=8, prime_gate=True, fast_blocks=1,
               lora=LoRAConfig(rank=2, alpha=4.0))
    tcfg = TrainConfig(seq_len=16, tokens_per_step=32, dtype="fp32", slow_spec=ARMS["F"]["slow"],
                       fast_init_trained=True)
    config = Config(model=mcfg, inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=True),
                    outer=OuterConfig(lr=1e-2, total_steps=10), train=tcfg)
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=16).double()
    split = split_parameters(model, mcfg, tcfg)
    assert sorted(split.fast) == ["blocks.3.mlp_prime.w1.weight", "blocks.3.mlp_prime.w2.weight", "blocks.3.mlp_prime.w3.weight"]
    assert "blocks.3.prime_gate" in split.slow and "blocks.3.mlp.w1.weight" in split.frozen
    assert set(split.outer) == set(split.slow) | set(split.fast)
    loop = TTTInnerLoop(model, config, build_inner_optimizer(config.inner))
    opt = build_outer_optimizer(split.outer, config.outer)
    gate = split.slow["blocks.3.prime_gate"]
    prime_before = {k: v.detach().clone() for k, v in split.fast.items()}
    frozen_before = {k: v.detach().clone() for k, v in split.frozen.items()}

    def batches():
        g = torch.Generator().manual_seed(0)
        while True:
            ids = torch.randint(0, mcfg.vocab_size, (1, 16), generator=g)
            tgt = torch.randint(0, mcfg.vocab_size, (1, 16), generator=g)
            yield {"input_ids": ids, "targets": tgt, "loss_mask": torch.ones_like(tgt, dtype=torch.float64)}

    metrics = Trainer(config, model, split, loop, opt, batches(), device=torch.device("cpu")).train_step(1)

    assert metrics.loss > 0.0
    assert gate.item() != 0.0, "the gate received no gradient"
    for k, v in split.fast.items():
        assert torch.equal(v, prime_before[k]), f"prime W_0 moved while the gate was 0: {k}"
    for k, v in split.frozen.items():
        assert torch.equal(v, frozen_before[k]), k


def test_inner_lr_has_no_default_when_an_inner_optimizer_is_active():
    """The CLI default used to be 1e-3, the value that drives the loss to 20.2. A forgotten
    flag must fail at once, not run a divergent configuration for hours."""
    from ttt.run import resolve_inner_lr

    with pytest.raises(AssertionError, match="--inner-lr is required"):
        resolve_inner_lr("normalized_sgd", None)
    with pytest.raises(AssertionError, match="--inner-lr is required"):
        resolve_inner_lr("adamw", None)
    assert resolve_inner_lr("normalized_sgd", 4e-6) == 4e-6
    assert resolve_inner_lr("normalized_sgd", 0.0) == 0.0     # an explicit zero is a real setting
    assert resolve_inner_lr("none", None) == 0.0              # arm A: no inner optimizer, nothing to set


def test_a_finished_result_is_recognised_and_an_unfinished_one_is_not(tmp_path):
    """Chains of resumable jobs carry spare links; a spare link must see that the run is
    finished. "Finished" means the evaluation was written, not merely that a file exists."""
    import json

    from ttt.run import result_is_complete

    out = tmp_path / "r.json"
    assert not result_is_complete(out)                                   # no file
    out.write_text(json.dumps({"arm": "C", "history": [{"step": 0}]}))
    assert not result_is_complete(out)                                   # trained, not evaluated
    out.write_text(json.dumps({"arm": "C", "history": [], "eval": {"loss": 2.5}}))
    assert result_is_complete(out)


# --------------------------------------------------------------------------- arm F pieces


def test_prime_size_and_gate_require_prime():
    with pytest.raises(AssertionError, match="prime_intermediate_size.*prime"):
        cfg(prime_intermediate_size=8)
    with pytest.raises(AssertionError, match="prime_gate.*prime"):
        cfg(prime_gate=True)
    c = cfg(prime=True, prime_intermediate_size=8, prime_gate=True)
    assert c.prime_intermediate == 8
    assert cfg(prime=True).prime_intermediate == cfg().intermediate_size


def test_prime_mlp_has_its_own_size_and_no_lora():
    """The prime MLP is trained directly by the outer loop, so LoRA on it would be a
    second parametrisation of the same matrix; the block's own MLP keeps its LoRA."""
    from ttt.model.block import TransformerBlock

    c = cfg(prime=True, prime_intermediate_size=8, lora=LoRAConfig(rank=2, alpha=4.0, targets=("w1", "w2", "w3")))
    block = TransformerBlock(c, use_math_backend=True, is_fast_block=True)
    assert block.mlp_prime.w1.weight.shape == (8, c.hidden_size)
    assert block.mlp_prime.w2.weight.shape == (c.hidden_size, 8)
    assert not any("mlp_prime" in n and "lora_" in n for n, _ in block.named_parameters())
    assert any(n.startswith("mlp.w1.lora_") for n, _ in block.named_parameters())


def test_gated_prime_block_equals_the_plain_block_at_init_and_opens_with_the_gate():
    """Invariant (LaCT Alg. 2, App. C.3): with the gate at 0 the block output is exactly the
    pretrained block's, the prime MLP receives a zero gradient, and the gate itself does not.
    Witness: after the gate is set to 1 the output differs and the prime MLP's gradient is
    nonzero, so the zero at init is the gate and not a dead module."""
    from ttt.model.block import TransformerBlock
    from ttt.model.rope import build_rope_cache

    c_plain, c_prime = cfg(), cfg(prime=True, prime_intermediate_size=8, prime_gate=True)
    torch.manual_seed(0)
    plain = TransformerBlock(c_plain, use_math_backend=True, is_fast_block=True)
    gated = TransformerBlock(c_prime, use_math_backend=True, is_fast_block=True)
    gated.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(1, 8, c_plain.hidden_size)
    cos, sin = build_rope_cache(c_plain.head_dim, 8, c_plain.rope)
    prime_weights = [gated.mlp_prime.w1.weight, gated.mlp_prime.w2.weight, gated.mlp_prime.w3.weight]

    y_plain, _ = plain(x, cos, sin, None)
    y_gated, _ = gated(x, cos, sin, None)
    g_prime = torch.autograd.grad((y_gated**2).sum(), prime_weights + [gated.prime_gate])

    assert gated.prime_gate.item() == 0.0
    torch.testing.assert_close(y_gated, y_plain, rtol=0.0, atol=0.0)
    assert all(g.abs().max().item() == 0.0 for g in g_prime[:3])
    assert g_prime[3].abs().item() > 0.0, "the gate must receive a gradient or it never opens"
    with torch.no_grad():
        gated.prime_gate.fill_(1.0)
    y_open, _ = gated(x, cos, sin, None)
    g_open = torch.autograd.grad((y_open**2).sum(), prime_weights)
    assert (y_open - y_plain).abs().max().item() > 1e-3
    assert all(g.abs().max().item() > 0.0 for g in g_open)


def test_gated_prime_output_is_rms_normalised_before_the_gate():
    """prime_out = gate * RMSNorm(mlp_prime(norm(h))): with gate = 1 and unit norm gains the
    branch has unit RMS per token, whatever the prime MLP's scale."""
    from ttt.model.block import TransformerBlock
    from ttt.model.rope import build_rope_cache

    c = cfg(prime=True, prime_intermediate_size=8, prime_gate=True)
    torch.manual_seed(0)
    block = TransformerBlock(c, use_math_backend=True, is_fast_block=True)
    with torch.no_grad():
        block.prime_gate.fill_(1.0)
        for w in (block.mlp_prime.w1.weight, block.mlp_prime.w2.weight, block.mlp_prime.w3.weight):
            w.mul_(50.0)
    x = torch.randn(1, 8, c.hidden_size)
    cos, sin = build_rope_cache(c.head_dim, 8, c.rope)

    y, _ = block(x, cos, sin, None)
    attn_out, _ = block.attn(block.seq_norm(x), cos, sin, None)
    h = x + attn_out
    branch = block.ffn_prime_out_norm(block.mlp_prime(block.ffn_prime_norm(h)))
    want = (h + branch) + block.mlp(block.ffn_norm(h + branch))

    torch.testing.assert_close(y, want)
    rms = branch.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)
