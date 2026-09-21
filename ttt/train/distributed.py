"""Data parallelism over sequences, by hand.

Why not torch's DistributedDataParallel: the outer gradient here is accumulated by several
`backward()` calls per sequence (one per truncation window) through `functional_call`, and
DDP's hooks assume one backward per forward through the wrapped module. The structure of
the problem makes a wrapper unnecessary anyway:

    fast weights are per-sequence  =>  the S sequences of an outer step are independent
    =>  g = (1/S) sum_{i<S} g_i  =  sum_{r<R} [ (1/S) sum_{i in shard r} g_i ]

So every rank accumulates its shard with the GLOBAL factor 1/S, and ONE all-reduce SUM of the
slow-weight gradients per step yields exactly the single-process gradient. All ranks then
clip and step identically, so their weights stay equal without ever being broadcast.
Only the slow weights (tens of millions of parameters) cross the wire.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class Dist:
    """This process's place in the job. world_size == 1 means no process group exists."""

    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        assert self.world_size >= 1 and 0 <= self.rank < self.world_size, (self.rank, self.world_size)

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_from_env(device_type: str) -> tuple[Dist, int]:
    """Join the process group described by the environment. Returns (Dist, local_rank).

    Under `srun` Slurm sets SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID; `torchrun` sets
    RANK / WORLD_SIZE / LOCAL_RANK. With neither (or one task) this is a single process and
    no group is created. MASTER_ADDR and MASTER_PORT must be set by the launcher.
    """
    env = os.environ
    world = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1")))
    if world == 1:
        return Dist(), 0
    rank = int(env["RANK"] if "RANK" in env else env["SLURM_PROCID"])
    local = int(env["LOCAL_RANK"] if "LOCAL_RANK" in env else env["SLURM_LOCALID"])
    assert "MASTER_ADDR" in env and "MASTER_PORT" in env, "MASTER_ADDR / MASTER_PORT must be set for multi-process runs"
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=world)
    return Dist(rank=rank, world_size=world), local


def all_reduce_sum_grads_(params: list[Tensor], d: Dist) -> None:
    """In place: grad <- sum over ranks of grad, for every parameter in `params`.

    One flat buffer, one collective. A parameter with no gradient on this rank contributes
    zeros: every rank must enter the collective with the same shape, and "no gradient here"
    means exactly a zero contribution to the sum.
    """
    if d.world_size == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n
    assert offset == flat.numel()


def all_reduce_sum_scalar(value: float, d: Dist, device: torch.device) -> float:
    """Sum of a Python float over ranks (used for the logged loss)."""
    if d.world_size == 1:
        return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t)
