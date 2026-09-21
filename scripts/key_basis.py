"""Measures the key directions that all tokens share, for `--inner preconditioned_sgd`.

For every fast matrix W [out, in] the script collects its inputs k_t [in] over TRAINING text, forms

    C = (1/N) sum_t k_t k_t^T          [in, in]   (uncentered, N tokens)

and keeps E [in, r], the r strongest eigenvectors of C (ttt/optim/key_basis.py). w1 and w3 of a block
read the same input and share one basis; w2 reads silu(w1 x) * w3 x. Test-time training is off while
keys are collected, so the basis describes the fast weights at their starting point W_0. With
--load-slow the keys are those of a trained model.

Takes every model option of `python -m ttt.run`:

    python scripts/key_basis.py --arm A --mode eval --data DIR --seq-len 32768 --chunk 1024 --window 1024 \\
        --prefix-segment 1024 --fast-blocks 4 --remat-group 1 --dtype bf16 --rank 64 --tokens 65536 \\
        --out results/key_basis_k1024_r64.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ttt.data.dataset import TokenSequenceDataset, _shard_indices
from ttt.optim.key_basis import save_key_basis
from ttt.run import build_everything, build_parser
from ttt.train.checkpoint import load_slow_weights


class SecondMoment:
    """Running sum of k k^T in float64, fed with [..., in] activations."""

    def __init__(self) -> None:
        self.total: Tensor | None = None  # [in, in]
        self.count = 0

    def add(self, keys: Tensor) -> None:
        k = keys.detach().reshape(-1, keys.shape[-1]).float()  # [tokens, in]
        product = (k.mT @ k).double()
        self.total = product if self.total is None else self.total + product
        self.count += k.shape[0]

    def strongest_directions(self, rank: int) -> tuple[Tensor, float, list[float]]:
        """(E [in, rank] by falling eigenvalue, share of trace(C) they carry, those eigenvalues / trace)."""
        assert self.total is not None and self.count > 0, "no keys were collected"
        values, vectors = torch.linalg.eigh(self.total / self.count)  # ascending
        top_values = values[-rank:].flip(0)
        trace = float(values.sum())
        return vectors[:, -rank:].flip(1).float(), float(top_values.sum()) / trace, (top_values / trace).tolist()


def main() -> None:
    p = build_parser()
    p.add_argument("--rank", type=int, required=True, help="r: number of shared key directions kept per matrix")
    p.add_argument("--tokens", type=int, default=65536, help="calibration tokens from the TRAINING split; a multiple of --seq-len")
    args = p.parse_args()
    assert args.mode == "eval", "key statistics are evaluation only: pass --mode eval"
    assert args.tokens > 0 and args.tokens % args.seq_len == 0, f"--tokens {args.tokens} must be a positive multiple of --seq-len {args.seq_len}"

    torch.manual_seed(args.seed)
    cfg, model, split, loop, device = build_everything(args)
    assert cfg.inner.optimizer == "none", (
        f"keys are collected with test-time training off, got --inner {cfg.inner.optimizer}; pass --arm A, or "
        "--inner none together with --load-slow for trained weights"
    )
    loaded = load_slow_weights(Path(args.load_slow), split=split) if args.load_slow else None

    fast_layers = range(cfg.model.first_fast_layer, cfg.model.num_layers)
    moments: dict[tuple[int, str], SecondMoment] = {}
    handles = []
    for layer in fast_layers:
        mlp = getattr(model.blocks[layer], cfg.model.fast_module)
        for role, module in (("in", mlp), ("w2", mlp.w2)):
            moment = moments[(layer, role)] = SecondMoment()
            handles.append(module.register_forward_pre_hook(lambda _module, inputs, moment=moment: moment.add(inputs[0])))

    ds = TokenSequenceDataset(Path(args.data), "train", args.seq_len)
    order = _shard_indices(len(ds), shuffle=True, seed=args.seed, rank=0, world_size=1)[: args.tokens // args.seq_len]
    for n, index in enumerate(order):
        item = ds[index]
        ids, tgt, mask = (item[key].unsqueeze(0).to(device) for key in ("input_ids", "targets", "loss_mask"))
        with torch.enable_grad():  # run_sequence's contract, although nothing is differentiated here
            loop.run_sequence(ids, tgt, mask, dict(split.fast), inference=True)
        print(f"[{n + 1}/{len(order)}] sequence {index}", flush=True)
    for handle in handles:
        handle.remove()

    basis: dict[str, Tensor] = {}
    shares: dict[str, float] = {}
    spectra: dict[str, list[float]] = {}
    for (layer, role), moment in sorted(moments.items()):
        # Each hook must have seen every calibration token exactly once (a recomputed block would double it).
        assert moment.count == args.tokens, f"block {layer} {role}: collected {moment.count} keys for {args.tokens} tokens"
        directions, share, spectrum = moment.strongest_directions(args.rank)
        prefix = f"blocks.{layer}.{cfg.model.fast_module}"
        for matrix in (("w1", "w3") if role == "in" else ("w2",)):
            name = f"{prefix}.{matrix}.weight"
            basis[name], shares[name], spectra[name] = directions, share, spectrum[:8]
        print(f"block {layer} {'w1, w3' if role == 'in' else 'w2':<6} [in = {directions.shape[0]}]: the {args.rank} strongest "
              f"directions carry {share:.1%} of trace(C); the first alone {spectrum[0]:.1%}", flush=True)
    assert set(basis) == set(split.fast), f"basis names {sorted(set(basis) ^ set(split.fast))} do not match the fast weights"

    meta = {"args": {k: v for k, v in sorted(vars(args).items())}, "tokens": args.tokens, "rank": args.rank,
            "sequence_indices": [int(i) for i in order], "trace_share": shares, "leading_eigenvalue_shares": spectra,
            "loaded_slow_step": None if loaded is None else loaded["step"]}
    save_key_basis(Path(args.out), basis, meta)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
