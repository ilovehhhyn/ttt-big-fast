"""Data-parallel training must be the SAME optimisation as one process doing all the work.

The 32 sequences of an outer step are independent (fast weights are per-sequence), so the
step's gradient is a sum over sequences and splits across ranks exactly:

    g = (1/S) sum_{i<S} g_i  =  sum_{r<R}  (1/S) sum_{i in shard r} g_i

Each rank accumulates its shard with the GLOBAL 1/S, and an all-reduce SUM over ranks gives g.
The test runs two real processes (gloo, CPU, float64) against one process, on the same
sequence set, and demands the same slow weights on every rank after several steps.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, OuterConfig, RopeConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.optim.outer import build_outer_optimizer
from ttt.train.distributed import Dist
from ttt.train.inner_loop import TTTInnerLoop
from ttt.train.trainer import Trainer

STEPS = range(1, 4)  # step 0 has outer lr exactly 0, so it would test nothing
GLOBAL_SEQS = 4


def build(truncate_bptt: int):
    mcfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
                       num_heads=4, num_kv_heads=2, window_size=8, chunk_size=4, fast_blocks=1,
                       rope=RopeConfig(theta=10000.0, scaling="none"), lora=LoRAConfig(rank=2, alpha=4.0))
    cfg = Config(model=mcfg,
                 inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=True),
                 outer=OuterConfig(lr=1e-2, total_steps=10),
                 train=TrainConfig(seq_len=16, tokens_per_step=16 * GLOBAL_SEQS, dtype="fp32",
                                   truncate_bptt=truncate_bptt))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=16).double()
    split = split_parameters(model, mcfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    return cfg, model, split, loop, build_outer_optimizer(split.slow, cfg.outer)


def stream(cfg, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
        tgt = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
        yield {"input_ids": ids, "targets": tgt, "loss_mask": torch.ones_like(tgt, dtype=torch.float64)}


def _train(truncate_bptt: int, d: Dist) -> tuple[dict, list[dict]]:
    cfg, model, split, loop, opt = build(truncate_bptt)
    # Rank r reads items r, r + R, r + 2R, ... : the same sharding build_dataloader uses, so the
    # union over ranks of each step's items is exactly the single-process step's items.
    shard = itertools.islice(stream(cfg), d.rank, None, d.world_size)
    tr = Trainer(cfg, model, split, loop, opt, shard, device=torch.device("cpu"), dist=d)
    logs = [tr.train_step(s).as_log() for s in STEPS]
    return {k: v.detach().clone() for k, v in split.slow.items()}, logs


def _worker(rank: int, world_size: int, truncate_bptt: int, port: int, out: dict) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        slow, logs = _train(truncate_bptt, Dist(rank=rank, world_size=world_size))
        out[rank] = ({k: v.numpy() for k, v in slow.items()}, [(m["loss"], m["grad_norm"]) for m in logs])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("truncate_bptt", [0, 2])
def test_two_ranks_equal_one_process(truncate_bptt: int) -> None:
    ref_slow, ref_logs = _train(truncate_bptt, Dist(rank=0, world_size=1))

    port = 29500 + truncate_bptt + (hash(str(truncate_bptt)) % 97)
    with mp.Manager() as manager:
        out = manager.dict()
        mp.spawn(_worker, args=(2, truncate_bptt, port, out), nprocs=2, join=True)
        results = dict(out)

    assert set(results) == {0, 1}
    for rank, (slow, logs) in results.items():
        for k, v in ref_slow.items():
            got = torch.from_numpy(slow[k])
            assert torch.allclose(got, v, rtol=0, atol=1e-12), f"rank {rank}: {k} differs by {(got - v).abs().max():.2e}"
        # The logged loss and gradient norm are GLOBAL quantities, identical on every rank.
        for (loss, gnorm), ref in zip(logs, ref_logs, strict=True):
            assert abs(loss - ref["loss"]) < 1e-12 and abs(gnorm - ref["grad_norm"]) < 1e-10, (rank, loss, ref)


def test_sequences_per_step_must_divide_across_ranks() -> None:
    cfg, model, split, loop, opt = build(0)
    with pytest.raises(AssertionError, match="must be divisible by world_size"):
        Trainer(cfg, model, split, loop, opt, iter(()), device=torch.device("cpu"), dist=Dist(rank=0, world_size=3))
