"""Load TTT-E2E (arm E) weights from a JAX/Equinox orbax checkpoint into TTTTransformer.

The authors' own checkpoints live in a requester-pays GCS bucket, so we use a free
third-party reproduction on Hugging Face, `Luxel/ttt-e2e-760m-results`. Any number
derived from it is labelled "third-party reproduction" in the results table.

THE SOURCE PYTREE. The reference implementation (github.com/test-time-training/e2e)
is JAX + Equinox. Equinox modules are pytrees, so orbax stores them as a nested dict
keyed by attribute name. Two things follow that have no analogue in `hf_import`:

1. Blocks are built with `jax.vmap(lambda k: Block(config, key=k))(keys)`, so every
   per-layer array is STACKED along a leading axis: one entry of shape
   [num_layers, ...] rather than `num_layers` separate entries. The prime tensors
   live in `prime_storage` and are stacked over `suffix_len` instead, and
   `BlockCollectionSplit` slices them onto the LAST `suffix_len` blocks, so
   prime index j is our block `first_fast_layer + j`.
2. `NormalLinear.weight` has shape [in_features, out_features] and is applied as
   `x @ weight`; `torch.nn.Linear.weight` is [out_features, in_features] and is
   applied as `x @ weight.T`. Every linear must therefore be TRANSPOSED. RMSNorm
   scales and the embedding table are not.

Name mapping (e2e -> ours), with `P = "language_model.model."`:

    P + wte.weight                                  -> embed_tokens.weight
    P + ln_f.weight                                 -> norm.weight
    P + h.blocks.seq_norm.weight             [L, d] -> blocks.{i}.seq_norm.weight
    P + h.blocks.ffn_norm.weight             [L, d] -> blocks.{i}.ffn_norm.weight
    P + h.blocks.seq_post_norm.weight        [L, d] -> blocks.{i}.seq_post_norm.weight
    P + h.blocks.ffn_post_norm.weight        [L, d] -> blocks.{i}.ffn_post_norm.weight
    P + h.blocks.seq_modeling_block.wq.w  [L, d, d] -> blocks.{i}.attn.wq.weight     (T)
    P + h.blocks.seq_modeling_block.wk.w  [L, d, d] -> blocks.{i}.attn.wk.weight     (T)
    P + h.blocks.seq_modeling_block.wv.w  [L, d, d] -> blocks.{i}.attn.wv.weight     (T)
    P + h.blocks.seq_modeling_block.wo.w  [L, d, d] -> blocks.{i}.attn.wo.weight     (T)
    P + h.blocks.seq_modeling_block.q_norm.w[L, hd] -> blocks.{i}.attn.q_norm.weight
    P + h.blocks.seq_modeling_block.k_norm.w[L, hd] -> blocks.{i}.attn.k_norm.weight
    P + h.blocks.feed_forward.w1.weight   [L, d, f] -> blocks.{i}.mlp.w1.weight      (T)
    P + h.blocks.feed_forward.w2.weight   [L, f, d] -> blocks.{i}.mlp.w2.weight      (T)
    P + h.blocks.feed_forward.w3.weight   [L, d, f] -> blocks.{i}.mlp.w3.weight      (T)
    P + h.prime_storage.ffn_prime_norm.w     [S, d] -> blocks.{18+j}.ffn_prime_norm.weight
    P + h.prime_storage.ffn_prime_post_norm.w[S, d] -> blocks.{18+j}.ffn_prime_post_norm.weight
    P + h.prime_storage.feed_forward_prime.w1 [S,d,f] -> blocks.{18+j}.mlp_prime.w1.weight (T)
    P + h.prime_storage.feed_forward_prime.w2 [S,f,d] -> blocks.{18+j}.mlp_prime.w2.weight (T)
    P + h.prime_storage.feed_forward_prime.w3 [S,d,f] -> blocks.{18+j}.mlp_prime.w3.weight (T)

`language_model.lm_head` is a None leaf because `tie_word_embeddings` is true, as are
every `bias` (all RMSNorms use `use_bias=False`), the per-block `ffn_prime_*` slots
(the prime modules live in `prime_storage`) and the `step_index` / `kv_cache_index` /
`chunk_index` counters. None leaves carry no tensor and are dropped by the reader.

Nothing is skipped silently: a key we expect but do not find, and a tensor we find but
do not consume, both raise.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from ttt.config import LoRAConfig, ModelConfig, RopeConfig
from ttt.model.transformer import TTTTransformer

E2E_REPO = "Luxel/ttt-e2e-760m-results"

#: short stage name -> path prefix inside the repo. Each holds
#: `checkpoint/<STEP>/model_weights` and `experiment/resolved_config.yaml`.
STAGES: dict[str, str] = {
    "s2_adapt": "protocol_r_760m_author_seed_v1/stages/S2_ADAPT/adapt-760m-e2e-8K-from-fa",
    "s2_bridge": "protocol_r_760m_author_seed_v1/stages/S2/ext-760m-e2e-32K-from-fa-bridge",
    "s3": "protocol_r_760m_author_seed_v1/stages/S3/ext-760m-e2e-32K",
}

SRC_PREFIX = "language_model.model."
BLOCKS_PREFIX = SRC_PREFIX + "h.blocks."
PRIME_PREFIX = SRC_PREFIX + "h.prime_storage."

# e2e name -> our name. Neither stacked nor transposed.
GLOBAL_KEYS: dict[str, str] = {
    SRC_PREFIX + "wte.weight": "embed_tokens.weight",
    SRC_PREFIX + "ln_f.weight": "norm.weight",
}
# Stacked over num_layers. suffix under BLOCKS_PREFIX -> suffix under blocks.{i}.
BLOCK_NORMS: dict[str, str] = {
    "seq_norm.weight": "seq_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "seq_post_norm.weight": "seq_post_norm.weight",
    "ffn_post_norm.weight": "ffn_post_norm.weight",
    "seq_modeling_block.q_norm.weight": "attn.q_norm.weight",
    "seq_modeling_block.k_norm.weight": "attn.k_norm.weight",
}
BLOCK_LINEARS: dict[str, str] = {
    "seq_modeling_block.wq.weight": "attn.wq.weight",
    "seq_modeling_block.wk.weight": "attn.wk.weight",
    "seq_modeling_block.wv.weight": "attn.wv.weight",
    "seq_modeling_block.wo.weight": "attn.wo.weight",
    "feed_forward.w1.weight": "mlp.w1.weight",
    "feed_forward.w2.weight": "mlp.w2.weight",
    "feed_forward.w3.weight": "mlp.w3.weight",
}
# Stacked over suffix_len (== cfg.fast_blocks), landing on the trailing blocks.
PRIME_NORMS: dict[str, str] = {
    "ffn_prime_norm.weight": "ffn_prime_norm.weight",
    "ffn_prime_post_norm.weight": "ffn_prime_post_norm.weight",
}
PRIME_LINEARS: dict[str, str] = {
    "feed_forward_prime.w1.weight": "mlp_prime.w1.weight",
    "feed_forward_prime.w2.weight": "mlp_prime.w2.weight",
    "feed_forward_prime.w3.weight": "mlp_prime.w3.weight",
}


# ------------------------------------------------------------------------------ config


def e2e_760m_config(**overrides) -> ModelConfig:
    """Arm E's ModelConfig, transcribed from the stage's `experiment/resolved_config.yaml`.

    Verified against S2_ADAPT: 24 layers, hidden 1536, intermediate 3328, 16 heads,
    vocab 128256, tied embeddings, rms_norm_eps 1e-6, qk_norm / pre_norm / post_norm /
    prime all true, suffix_len 6, mini_batch_size 1024, sliding_window_size 8192,
    rope_theta 5e5 with NO rope scaling (their `precompute_freqs_cis` is plain RoPE, not
    llama3-scaled). `num_key_value_heads` is null: the e2e attention has no GQA, its
    wq/wk/wv are all hidden->hidden, so num_kv_heads == num_heads.

    `overrides` exists for tests (a 4-layer toy) and for experiment variants such as
    enabling LoRA; unknown field names raise rather than being ignored.
    """
    base = dict(
        num_layers=24, hidden_size=1536, intermediate_size=3328,
        num_heads=16, num_kv_heads=16, vocab_size=128256,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
        qk_norm=True, post_norm=True, prime=True,
        fast_blocks=6, window_size=8192, chunk_size=1024,
        rope=RopeConfig(theta=500000.0, scaling="none"),
        lora=LoRAConfig(rank=0),
    )
    known = {f.name for f in fields(ModelConfig)}
    unknown = sorted(set(overrides) - known)
    assert not unknown, f"unknown ModelConfig fields: {unknown}"
    base.update(overrides)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------- download


def _stage_prefix(stage: str) -> str:
    assert stage in STAGES, f"unknown stage {stage!r}; known stages: {sorted(STAGES)}"
    return STAGES[stage]


def available_steps(repo_id: str = E2E_REPO, stage: str = "s2_adapt") -> list[int]:
    """Step directories that actually contain a `model_weights` tree, ascending.

    Discovered from the repo file listing rather than hardcoded, because the three
    stages stop at different steps (S2_ADAPT at 23199, the 32K stages at 11599)."""
    from huggingface_hub import list_repo_files

    prefix = _stage_prefix(stage) + "/checkpoint/"
    steps = set()
    for name in list_repo_files(repo_id):
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix) :].split("/")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] == "model_weights":
            steps.add(int(parts[0]))
    assert steps, f"no checkpoint steps under {prefix!r} in {repo_id}"
    return sorted(steps)


def download_checkpoint(repo_id: str = E2E_REPO, stage: str = "s2_adapt", *,
                        step: int | None = None, cache_dir: str | None = None) -> Path:
    """Snapshot-download just that stage's `model_weights` tree. step=None -> largest step.

    The `opt_state` tree beside it is the outer optimiser's moments, which we never
    read, so it is excluded from the allow-pattern (it roughly triples the download).
    """
    from huggingface_hub import snapshot_download

    prefix = _stage_prefix(stage)
    steps = available_steps(repo_id, stage)
    if step is None:
        step = steps[-1]
    assert step in steps, f"step {step} not in {repo_id}:{stage}; available: {steps}"

    tree = f"{prefix}/checkpoint/{step}/model_weights"
    root = Path(snapshot_download(repo_id, cache_dir=cache_dir,
                                  allow_patterns=[f"{tree}/**", f"{prefix}/checkpoint/{step}/_*"]))
    path = root / tree
    assert path.is_dir(), f"expected {path} after snapshot_download"
    return path


# ------------------------------------------------------------------------------ reading


def _flatten_into(node, prefix: tuple[str, ...], out: dict[str, np.ndarray]) -> None:
    """Depth-first flatten of the restored pytree, dropping None leaves."""
    if node is None:
        return
    if isinstance(node, dict):
        for key, child in node.items():
            _flatten_into(child, prefix + (str(key),), out)
        return
    name = ".".join(prefix)
    assert name not in out, f"duplicate key {name!r} while flattening"
    out[name] = np.asarray(node)


def _read_ocdbt_with_tensorstore(path: Path) -> dict[str, np.ndarray]:
    """Fallback reader: open the ocdbt kvstore and read every zarr array in it.

    Orbax writes one zarr array per parameter, named by its dotted pytree path, into a
    single ocdbt store. The zarr version differs between orbax releases, so the codec is
    detected from the metadata key present (`zarr.json` = v3, `.zarray` = v2) rather
    than guessed.
    """
    try:
        import tensorstore as ts
    except ImportError as exc:  # the ONE documented fallback: orbax, else tensorstore.
        raise RuntimeError(
            f"cannot read {path}: neither `orbax-checkpoint` nor `tensorstore` is "
            "importable. Install one with `pip install orbax-checkpoint`."
        ) from exc

    base = {"driver": "ocdbt", "base": f"file://{path}/"}
    keys = [k.decode() for k in ts.KvStore.open(base).result().list().result()]
    codecs = {"zarr.json": "zarr3", ".zarray": "zarr"}
    arrays = {k.rpartition("/")[0]: codecs[k.rpartition("/")[2]]
              for k in keys if k.rpartition("/")[2] in codecs}
    assert arrays, f"no zarr arrays in the ocdbt store at {path}; first keys: {sorted(keys)[:8]}"

    out: dict[str, np.ndarray] = {}
    for name, driver in sorted(arrays.items()):
        store = ts.open({"driver": driver, "kvstore": {**base, "path": name}}, open=True).result()
        out[name] = np.asarray(store.read().result())
    return out


def read_orbax_tree(path: Path) -> dict[str, np.ndarray]:
    """Flatten the orbax checkpoint to {dotted_key: array}.

    Uses `orbax.checkpoint` when importable, else reads the tensorstore/ocdbt store
    directly; raises a clear error if neither is available. Arrays are restored as
    numpy, never as `jax.Array`, so the conversion needs no JAX devices and does not
    have to reproduce the 8-way sharding the checkpoint was written with.
    """
    path = Path(path)
    assert path.is_dir(), f"not a checkpoint directory: {path}"
    try:
        import orbax.checkpoint as ocp
    except ImportError:
        return _read_ocdbt_with_tensorstore(path)

    import jax

    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(path)
    structure = metadata.item_metadata.tree if hasattr(metadata, "item_metadata") else metadata
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), structure
    )
    restored = checkpointer.restore(path, restore_args=restore_args)
    out: dict[str, np.ndarray] = {}
    _flatten_into(restored, (), out)
    return out


# ------------------------------------------------------------------------------ mapping


def expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter the import must produce, with the shape cfg implies.

    This is the contract `map_e2e_to_ours` asserts against, derived from cfg alone so
    that a wrong transpose or a wrong unstack cannot pass unnoticed."""
    d, f, hd, L = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_layers
    q_dim, kv_dim = cfg.num_heads * hd, cfg.num_kv_heads * hd
    exp: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (cfg.vocab_size, d),
        "norm.weight": (d,),
    }
    for i in range(L):
        b = f"blocks.{i}."
        for n in ("seq_norm", "ffn_norm", "seq_post_norm", "ffn_post_norm"):
            exp[b + n + ".weight"] = (d,)
        exp[b + "attn.wq.weight"] = (q_dim, d)
        exp[b + "attn.wk.weight"] = (kv_dim, d)
        exp[b + "attn.wv.weight"] = (kv_dim, d)
        exp[b + "attn.wo.weight"] = (d, q_dim)
        exp[b + "attn.q_norm.weight"] = (hd,)
        exp[b + "attn.k_norm.weight"] = (hd,)
        exp[b + "mlp.w1.weight"] = (f, d)
        exp[b + "mlp.w2.weight"] = (d, f)
        exp[b + "mlp.w3.weight"] = (f, d)
        if i >= cfg.first_fast_layer:
            exp[b + "ffn_prime_norm.weight"] = (d,)
            exp[b + "ffn_prime_post_norm.weight"] = (d,)
            exp[b + "mlp_prime.w1.weight"] = (f, d)
            exp[b + "mlp_prime.w2.weight"] = (d, f)
            exp[b + "mlp_prime.w3.weight"] = (f, d)
    return exp


def _to_torch(array: np.ndarray) -> torch.Tensor:
    """Float32 contiguous torch tensor. bf16 checkpoints are widened here rather than
    at copy time so that `load_e2e_into_model` sees one dtype."""
    array = np.asarray(array)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(array))


def map_e2e_to_ours(tree: dict[str, np.ndarray], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Unstack the vmapped leading axis, transpose linear weights, and rename.

    Every key we expect must be present (else KeyError naming it, plus the keys that
    ARE there) and every source tensor must be consumed (else KeyError naming the
    leftovers). Every produced tensor's shape is asserted against `expected_shapes`.
    """
    assert cfg.qk_norm and cfg.post_norm and cfg.prime, (
        "an e2e checkpoint always carries qk_norm, post_norm and prime tensors; "
        f"got qk_norm={cfg.qk_norm}, post_norm={cfg.post_norm}, prime={cfg.prime}"
    )
    assert cfg.num_kv_heads == cfg.num_heads, (
        f"e2e attention has no GQA (wq/wk/wv are all hidden->hidden), but cfg has "
        f"num_heads={cfg.num_heads}, num_kv_heads={cfg.num_kv_heads}"
    )
    assert 0 < cfg.fast_blocks <= cfg.num_layers

    d, f, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    L, S, first = cfg.num_layers, cfg.fast_blocks, cfg.first_fast_layer
    unconsumed = dict(tree)

    def take(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in unconsumed:
            raise KeyError(
                f"orbax checkpoint is missing {name!r}. Keys present: {sorted(tree)}"
            )
        array = np.asarray(unconsumed.pop(name))
        assert array.shape == shape, (
            f"{name}: checkpoint shape {array.shape} != {shape} implied by the config"
        )
        return array

    out: dict[str, torch.Tensor] = {}
    for src, dst in GLOBAL_KEYS.items():
        out[dst] = _to_torch(take(src, expected_shapes(cfg)[dst]))

    # Per-layer tensors: one stacked array, sliced into num_layers blocks.
    for src, dst in BLOCK_NORMS.items():
        # q_norm/k_norm are per-head (RMSNorm over head_dim); the rest are over hidden.
        size = hd if src.startswith("seq_modeling_block.") else d
        stacked = take(BLOCKS_PREFIX + src, (L, size))
        for i in range(L):
            out[f"blocks.{i}.{dst}"] = _to_torch(stacked[i])
    for src, dst in BLOCK_LINEARS.items():
        in_f, out_f = (d, f) if src.endswith(("w1.weight", "w3.weight")) else (f, d)
        if src.startswith("seq_modeling_block."):
            in_f = out_f = d  # wq/wk/wv/wo are all hidden -> hidden
        stacked = take(BLOCKS_PREFIX + src, (L, in_f, out_f))
        for i in range(L):
            out[f"blocks.{i}.{dst}"] = _to_torch(stacked[i].T)  # [in, out] -> [out, in]

    # Prime tensors: stacked over suffix_len, sliced onto the TRAILING blocks
    # (e2e's BlockCollectionSplit takes blocks[-suffix_len:]).
    for src, dst in PRIME_NORMS.items():
        stacked = take(PRIME_PREFIX + src, (S, d))
        for j in range(S):
            out[f"blocks.{first + j}.{dst}"] = _to_torch(stacked[j])
    for src, dst in PRIME_LINEARS.items():
        in_f, out_f = (d, f) if src.endswith(("w1.weight", "w3.weight")) else (f, d)
        stacked = take(PRIME_PREFIX + src, (S, in_f, out_f))
        for j in range(S):
            out[f"blocks.{first + j}.{dst}"] = _to_torch(stacked[j].T)

    if unconsumed:
        raise KeyError(
            f"unconsumed tensors in the checkpoint (nothing may be silently skipped): "
            f"{sorted(unconsumed)}"
        )

    exp = expected_shapes(cfg)
    assert set(out) == set(exp), (
        f"unproduced: {sorted(set(exp) - set(out))} | unexpected: {sorted(set(out) - set(exp))}"
    )
    for name, tensor in out.items():
        assert tuple(tensor.shape) == exp[name], (
            f"{name}: produced {tuple(tensor.shape)}, config implies {exp[name]}"
        )
    return out


# ------------------------------------------------------------------------------ loading


def _pretrained_param_names(model: TTTTransformer) -> set[str]:
    """Parameters that must receive a checkpoint tensor.

    LoRA adapters (B=0, A random) and the learned inner-LR log scalars (init 0) are new
    slow parameters with no e2e counterpart, so they are excluded by design."""
    return {
        name for name, _ in model.named_parameters()
        if "lora_" not in name and not name.startswith("inner_lr_log.")
    }


def load_e2e_into_model(model: TTTTransformer, tree: dict[str, np.ndarray],
                        cfg: ModelConfig) -> None:
    """Convert `tree` and copy it into `model` in place, asserting coverage both ways."""
    mapped = map_e2e_to_ours(tree, cfg)
    own = dict(model.named_parameters())
    targets = _pretrained_param_names(model)
    assert set(mapped) == targets, (
        f"unmapped model parameters: {sorted(targets - set(mapped))} | "
        f"mapped tensors with no parameter: {sorted(set(mapped) - targets)}"
    )
    for name in sorted(mapped):
        src, dst = mapped[name], own[name]
        assert tuple(src.shape) == tuple(dst.shape), (
            f"shape mismatch for {name}: checkpoint gives {tuple(src.shape)}, "
            f"model parameter is {tuple(dst.shape)}"
        )
        with torch.no_grad():
            dst.copy_(src.to(dst.dtype))


def build_e2e_ttt(stage: str = "s2_adapt", *, max_seq_len: int,
                  cache_dir: str | None = None,
                  dtype: torch.dtype = torch.float32) -> TTTTransformer:
    """Fully-loaded TTTTransformer with the third-party TTT-E2E 760M reproduction weights."""
    path = download_checkpoint(E2E_REPO, stage, cache_dir=cache_dir)
    tree = read_orbax_tree(path)
    cfg = e2e_760m_config()
    model = TTTTransformer(cfg, max_seq_len=max_seq_len).to(dtype)
    load_e2e_into_model(model, tree, cfg)
    return model


# ---------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    """Download + convert + `torch.save`, so the cluster job needs no network access."""
    p = argparse.ArgumentParser(prog="python -m ttt.utils.orbax_import", description=__doc__.split("\n")[0])
    p.add_argument("--repo", default=E2E_REPO)
    p.add_argument("--stage", default="s2_adapt", choices=sorted(STAGES))
    p.add_argument("--step", type=int, default=None, help="default: the largest step present")
    p.add_argument("--out", required=True, type=Path, help="destination .pt for the state dict")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args(argv)

    path = download_checkpoint(args.repo, args.stage, step=args.step, cache_dir=args.cache_dir)
    print(f"checkpoint: {path}")
    tree = read_orbax_tree(path)
    print(f"read {len(tree)} source tensors")
    cfg = e2e_760m_config()
    state = map_e2e_to_ours(tree, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    total = sum(t.numel() for t in state.values())
    print(f"wrote {len(state)} tensors ({total / 1e6:.1f}M parameters) to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
