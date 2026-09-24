"""Experiment runner.

Arms (plan section 0.1). All share the same model, data and evaluation code; they
differ only in which parameters are fast, which are slow, and whether the inner
loop takes a step.

    A  baseline   fast MLPs present but inner optimizer = none  -> no TTT at all.
                  Using the same chunked path as arm C (rather than a plain forward)
                  keeps the compute graph identical, so A vs C isolates TTT itself.
    B  ttt-naive  inner loop on, nothing slow -> dynamic evaluation, no meta-learning.
    C  proposed   fast = MLPs of the last `fast_blocks`; slow = LoRA (by default on the
                  attention projections AND the MLPs, see --lora-targets) + RMSNorm gains
                  + learned per-tensor inner LRs.
    D  full-slow  same fast set, every parameter slow.
    F  paper-layout control: static pretrained MLP kept as safe storage, a separate prime MLP
       carries the fast weights (plan section 0.1 and Task 7). NOT IMPLEMENTED: refuses to run.

Usage:
    python -m ttt.run --arm A --mode eval  --data DIR --out results/A.json
    python -m ttt.run --arm C --mode train --data DIR --out results/C.json --steps 250
"""

from __future__ import annotations


import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ttt.config import Config, InnerConfig, LoRAConfig, OuterConfig, TrainConfig
from ttt.data.dataset import build_dataloader
from ttt.eval.evaluator import evaluate
from ttt.eval.forgetting import build_probe_batch
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.optim.outer import build_outer_optimizer
from ttt.train.checkpoint import load_checkpoint, load_slow_weights, save_checkpoint, training_fingerprint
from ttt.train.distributed import init_from_env
from ttt.train.inner_loop import TTTInnerLoop
from ttt.train.trainer import Trainer
from ttt.utils.hf_import import MIRROR_REPO, build_llama_ttt

ARMS = {
    # Arm E is the reference TTT-E2E model: a different architecture (24 layers, prime
    # MLPs, qk-norm, post-norm) loaded from a converted orbax checkpoint, scored on our
    # split. It is evaluation-only and uses e2e's exact inner rule.
    "E": dict(inner="clipped_sgd", lora_rank=0, slow=(), prime=False),
    "A": dict(inner="none", lora_rank=0, slow=(), prime=False),
    "B": dict(inner="normalized_sgd", lora_rank=0, slow=(), prime=False),
    # Arm C's slow set includes LoRA on the MLP (w1,w2,w3), i.e. on the fast weights
    # themselves. That is NOT redundant: in TTT-E2E the fast-weight INITIALISATION W0 is
    # the single most important slow parameter (their outer loop optimises it directly).
    # We cannot meta-learn all of W0 from a frozen pretrained model, but a rank-r shift of
    # it is exactly the mechanism that makes W0 a good starting point for test-time
    # updates. Omitting it removes the main lever. Attention LoRA additionally shapes what
    # gets written into the fast memory.
    "C": dict(inner="normalized_sgd", lora_rank=64, slow=("lora_A", "lora_B", "norm.weight", "inner_lr_log"), prime=False),
    "D": dict(inner="normalized_sgd", lora_rank=0, slow=("**",), prime=False),
    # F  paper layout: the pretrained MLP stays frozen as safe storage; a SMALL extra "prime" MLP
    #    (width --prime-intermediate, LaCT-style output RMSNorm and a zero-initialised gate) is
    #    the fast weight, its initial value W_0 is meta-learned (fast_init_trained), and the LoRA,
    #    norms, step sizes and gate are slow as in C.
    "F": dict(inner="normalized_sgd", lora_rank=64,
              slow=("lora_A", "lora_B", "norm.weight", "inner_lr_log", "prime_gate"), prime=True),
}


def result_is_complete(out: Path) -> bool:
    """True iff `out` already holds a finished result (it has an "eval" block).

    A run longer than any Slurm wall limit is submitted as a chain of resumable links, with
    a link or two to spare. Once the run has finished, a spare link must do nothing: without
    this check it would reload the final checkpoint and spend GPU time re-evaluating. A
    missing or unfinished file (no "eval" block yet) is not complete.
    """
    if not out.exists():
        return False
    return "eval" in json.loads(out.read_text())


def resolve_slow_spec(arm_slow: tuple[str, ...], *, token_rates: bool) -> tuple[str, ...]:
    """The arm's slow set, plus the per-token rates when they are on.

    The rates are meta-learned or they are nothing, so an arm with no slow set (A, B)
    refuses them rather than carrying an untrained module. Arm D's wildcard already
    covers every non-fast parameter.
    """
    if not token_rates:
        return arm_slow
    assert arm_slow, "--token-rates needs a slow set to learn the rates; use an arm with slow weights (C or D)"
    if arm_slow == ("**",):
        return arm_slow
    return arm_slow + ("token_rate",)


def resolve_prime_intermediate(arm: str, value: int | None) -> int | None:
    """Arm F's prime-MLP width, or None for every other arm.

    There is no default width: the fast set's size is the experiment, so arm F requires
    --prime-intermediate (2048 is the value chosen on 2026-09-23, a quarter of Llama's 8192),
    and any other arm refuses it because it would change nothing.
    """
    if ARMS[arm]["prime"]:
        assert value is not None and value > 0, (
            "--prime-intermediate is required with --arm F: the width of the extra fast MLP has no "
            "safe default (2048 was chosen on 2026-09-23)"
        )
        return value
    assert value is None, f"--prime-intermediate {value} has no effect with --arm {arm}; it belongs to --arm F"
    return None


def resolve_inner_lr(optimizer: str, inner_lr: float | None) -> float:
    """The inner step size, or a hard error if an active inner optimizer was given none.

    A default would be a trap: the only value that was ever the default (1e-3) diverges.
    With no inner optimizer the value is irrelevant and 0.0 is returned.
    """
    if optimizer == "none":
        return 0.0
    assert inner_lr is not None, (
        f"--inner-lr is required with --inner {optimizer}: there is no safe default. The unit is "
        "1/sqrt(n_fast) = 7.05e-5 for the 201M-parameter fast set; the measured 32K optimum is 4e-6."
    )
    return inner_lr


def build_everything(args) -> tuple[Config, torch.nn.Module, object, TTTInnerLoop, torch.device]:
    arm = ARMS[args.arm]
    prime_intermediate = resolve_prime_intermediate(args.arm, args.prime_intermediate)
    device = torch.device(args.device)
    if args.arm == "E":
        return _build_arm_e(args, arm, device)
    lora = LoRAConfig(rank=args.lora_rank if args.lora_rank is not None else arm["lora_rank"],
                      alpha=args.lora_alpha, scaling="rslora",
                      targets=tuple(args.lora_targets.split(",")))
    model = build_llama_ttt(args.repo, max_seq_len=args.seq_len, window_size=args.window,
                            chunk_size=args.chunk, fast_blocks=args.fast_blocks,
                            lora=lora if lora.rank > 0 else None,
                            dtype=torch.float32, cache_dir=args.hf_cache, token_rates=args.token_rates,
                            prime=arm["prime"], prime_intermediate_size=prime_intermediate, prime_gate=arm["prime"])
    # Master weights stay fp32; the forward runs under bf16 autocast (see
    # TTTInnerLoop._autocast). remat_blocks trades compute for the math-SDPA score
    # matrices, which dominate activation memory in the second-order path.
    model.remat_blocks = args.remat_blocks
    model = model.to(device)

    optimizer = arm["inner"] if args.inner is None else args.inner
    assert args.shared_keep is None or optimizer == "preconditioned_sgd", (
        f"--shared-keep {args.shared_keep} has no effect with --inner {optimizer}; it belongs to "
        "--inner preconditioned_sgd"
    )
    inner = InnerConfig(optimizer=optimizer,
                        lr_rms=resolve_inner_lr(optimizer, args.inner_lr), norm_scope=args.norm_scope,
                        ns_dtype=args.ns_dtype, weight_norm=args.weight_norm,
                        eps=args.adam_eps, clip_tau=args.clip_tau, beta1=0.9, beta2=0.9, warm_start=True,
                        learned_lr=bool(arm["slow"]) and "inner_lr_log" in arm["slow"],
                        delta_decay=args.delta_decay, key_basis_path=args.key_basis,
                        shared_keep=0.0 if args.shared_keep is None else args.shared_keep)
    outer = OuterConfig(lr=args.outer_lr, total_steps=args.steps)
    train = TrainConfig(seq_len=args.seq_len, tokens_per_step=args.tokens_per_step,
                        micro_batch=1, remat_group=args.remat_group,
                        prefix_segment=args.prefix_segment,
                        truncate_bptt=args.truncate_bptt,
                        slow_spec=resolve_slow_spec(arm["slow"], token_rates=args.token_rates) or ("__none__",),
                        fast_init_trained=arm["prime"], dtype=args.dtype)
    cfg = Config(model=model.cfg, inner=inner, outer=outer, train=train)
    split = split_parameters(model, model.cfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    return cfg, model, split, loop, device


def _build_arm_e(args, arm, device):
    """Arm E: the reference TTT-E2E 760M model from a converted orbax checkpoint."""
    from ttt.utils.orbax_import import e2e_760m_config

    assert args.e2e_ckpt, "--e2e-ckpt is required for arm E"
    mcfg = e2e_760m_config(window_size=args.window, chunk_size=args.chunk,
                           fast_blocks=args.e2e_fast_blocks)
    model = TTTTransformer(mcfg, max_seq_len=args.seq_len).to(torch.float32)
    state = torch.load(args.e2e_ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # inner_lr_log is ours, not the checkpoint's; nothing else may be missing.
    unexplained = [m for m in missing if not m.startswith("inner_lr_log.")]
    assert not unexplained, f"checkpoint is missing {len(unexplained)} tensors: {unexplained[:5]}"
    assert not unexpected, f"checkpoint has {len(unexpected)} unexpected tensors: {unexpected[:5]}"
    model = model.to(device)

    inner = InnerConfig(optimizer=args.inner or arm["inner"],
                        lr_rms=resolve_inner_lr(args.inner or arm["inner"], args.inner_lr),
                        ns_dtype=args.ns_dtype, weight_norm=args.weight_norm,
                        clip_tau=args.clip_tau, learned_lr=False)
    train = TrainConfig(seq_len=args.seq_len, tokens_per_step=args.tokens_per_step,
                        micro_batch=1, remat_group=args.remat_group,
                        prefix_segment=args.prefix_segment,
                        truncate_bptt=args.truncate_bptt,
                        slow_spec=("__none__",), dtype=args.dtype)
    # Eval-only: there is no training, so there is no warmup to schedule. Say so
    # explicitly -- the default fracs over total_steps=1 round to a 0-step warmup,
    # which is now (correctly) a hard error.
    inner = replace(inner, lr_warmup_frac=0.0)
    cfg = Config(model=mcfg, inner=inner,
                 outer=OuterConfig(lr=0.0, total_steps=1, warmup_frac=0.0), train=train)
    split = split_parameters(model, mcfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    return cfg, model, split, loop, device


def build_parser() -> argparse.ArgumentParser:
    """Every option that defines a model, its inner rule and its data. Shared with the
    evaluation-only scripts (scripts/recall_probe.py), so that they build exactly the model a
    training run built and cannot drift from it."""
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--mode", required=True, choices=["eval", "train"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=MIRROR_REPO)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--fast-blocks", type=int, default=4)
    p.add_argument("--remat-group", type=int, default=0)
    p.add_argument("--empty-cache", action="store_true",
                   help="release cached GPU blocks between sequences (slower, more headroom)")
    p.add_argument("--truncate-bptt", type=int, default=0,
                   help="differentiate every K chunks and release that window; meta-gradient spans "
                        "<=K inner steps, so it is BIASED but memory becomes O(K) not O(N) (0=exact)")
    p.add_argument("--prefix-segment", type=int, default=0,
                   help="segment the frozen prefix (0 = one shot); must divide seq_len and be <= window")
    p.add_argument("--remat-blocks", action="store_true",
                   help="recompute each suffix block during backward (big memory win)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--tokens-per-step", type=int, default=524288)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--inner", default=None,
                   choices=[None, "none", "normalized_sgd", "adamw", "muon", "clipped_sgd", "preconditioned_sgd"])
    p.add_argument("--key-basis", default=None,
                   help="preconditioned_sgd: the shared key directions of each fast matrix (scripts/key_basis.py)")
    p.add_argument("--shared-keep", type=float, default=None,
                   help="preconditioned_sgd: fraction in [0, 1] of the shared-direction part of the gradient "
                        "kept in the update (default 0 removes it; 1 is normalized_sgd exactly)")
    p.add_argument("--inner-lr", type=float, default=None,
                   help="per-element RMS of the inner step. REQUIRED whenever an inner optimizer is "
                        "active: there is no safe default. The unit is 1/sqrt(n_fast) (7.05e-5 for the "
                        "201M-parameter fast set); the measured 32K optimum is 4e-6. The old default, "
                        "1e-3, is 14x the unit and drives the loss to 20.2.")
    p.add_argument("--norm-scope", default="tensor", choices=["tensor", "global"])
    p.add_argument("--prime-intermediate", type=int, default=None,
                   help="arm F only, REQUIRED there: hidden width of the extra fast MLP (2048 chosen on "
                        "2026-09-23; Llama's own MLP is 8192)")
    p.add_argument("--token-rates", action="store_true",
                   help="per-token learning rates on the fast-weight write, eta_t = softplus(w.x_t + b), one "
                        "linear layer per fast block, meta-learned as a slow parameter (LaCT Eq. 4). "
                        "Init eta = 1, so a fresh run equals one without the flag. Needs a slow set (arm C or D)")
    p.add_argument("--weight-norm", default="none", choices=["none", "row_reset"],
                   help="after every inner step: row_reset rescales each row of a fast matrix to the row "
                        "norm it had before the step (LaCT Alg. 1 and 3); none uses the update as is")
    p.add_argument("--ns-dtype", default="float32", choices=["float32", "bfloat16"],
                   help="muon only: dtype of the five Newton-Schulz rounds (the fast weights, the gradient "
                        "and the update stay fp32). bfloat16 is the reference Muon's choice; measured 1.9%% "
                        "relative Frobenius difference from float32 on a 2048 x 8192 matrix")
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--clip-tau", type=float, default=1.0)
    p.add_argument("--e2e-ckpt", default=None, help="arm E: converted orbax state dict (.pt)")
    p.add_argument("--e2e-fast-blocks", type=int, default=6, help="arm E suffix_len (paper: 6)")
    p.add_argument("--delta-decay", type=float, default=0.0)
    p.add_argument("--outer-lr", type=float, default=1e-3)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", default="wq,wk,wv,wo,w1,w2,w3",
                   help="w1,w2,w3 put LoRA on the fast MLPs, meta-learning a rank-r shift "
                        "of the fast-weight initialisation W0")
    p.add_argument("--ckpt-every", type=int, default=1,
                   help="checkpoint every N outer steps (and always after the last). 1 suits arm C "
                        "(0.5 GB); arm D writes ~12 GB per checkpoint, so use a larger interval there. "
                        "A killed run loses at most N-1 steps of work, never correctness.")
    p.add_argument("--load-slow", default=None,
                   help="EVAL ONLY: evaluate the trained slow weights stored in this checkpoint, under "
                        "whatever inner rule this command specifies (not a resume: settings may differ)")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint file, written after every step and resumed from if it exists "
                        "(default: --out with a .ckpt suffix, so every training run is resumable)")
    p.add_argument("--eval-ttt-off", action="store_true",
                   help="also evaluate the SAME trained weights with the inner loop off, "
                        "isolating what test-time training contributes at inference")
    p.add_argument("--forgetting-probe-tokens", type=int, default=0,
                   help="score this many tokens of UNRELATED held-out text under the fast weights "
                        "left by each evaluated sequence and report NLL(W_T) - NLL(W_0); 0 = off")
    p.add_argument("--eval-sequences", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    assert args.ckpt_every >= 1, f"--ckpt-every must be >= 1, got {args.ckpt_every}"

    if result_is_complete(Path(args.out)):
        print(f"[done] {args.out} already holds a finished result; nothing to do. "
              f"(Delete it, or choose another --out, to run again.)", flush=True)
        return
    # Data parallelism over sequences (ttt/train/distributed.py): one process per GPU, launched
    # by srun. A single process is world_size 1 and creates no group.
    dist_info, local_rank = init_from_env("cuda" if args.device.startswith("cuda") else "cpu")
    if dist_info.world_size > 1:
        assert args.mode == "train", "multi-process runs are for training; evaluate with one process"
        if args.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
    main_rank = dist_info.is_main
    torch.manual_seed(args.seed)  # identical on every rank: they must build identical models
    cfg, model, split, loop, device = build_everything(args)
    counts = split.counts()
    if main_rank:
        print(f"[run] arm={args.arm} fast={counts['fast']:,} slow={counts['slow']:,} frozen={counts['frozen']:,} "
              f"outer={counts['outer']:,} fast_init_trained={split.fast_init_trained}", flush=True)
        print(f"[run] chunks={cfg.num_chunks} remat_group={loop.group} seqs_per_step={cfg.train.seqs_per_step} "
              f"world_size={dist_info.world_size}", flush=True)

    result = {"arm": args.arm, "mode": args.mode, "args": vars(args), "param_counts": counts,
              "num_chunks": cfg.num_chunks, "remat_group": loop.group}

    if args.load_slow:
        # Evaluating trained weights under a chosen inner rule. Training from them is a
        # resume and goes through --ckpt, which checks that the settings are unchanged.
        assert args.mode == "eval", "--load-slow is for --mode eval; use --ckpt to resume training"
        result["loaded_slow"] = load_slow_weights(Path(args.load_slow), split=split)
        print(f"[load-slow] {args.load_slow}: step {result['loaded_slow']['step']}", flush=True)

    if args.mode == "train":
        opt = build_outer_optimizer(split.outer, cfg.outer)
        # Resume. A job can die at any moment (wall limit, node failure), and on this
        # cluster a resubmission waits days, so every completed step is checkpointed and a
        # restart continues exactly where it stopped (see ttt/train/checkpoint.py).
        start_step, history = 0, []
        fingerprint = training_fingerprint(vars(args))
        # Which sequences a step sees depends on the sharding, so a resume must use the same
        # number of ranks. "device" is operational and already excluded (cuda:0 vs cuda:3).
        fingerprint["world_size"] = dist_info.world_size
        # Resumable BY DEFAULT: the path is derived from --out, so a run is recoverable
        # even when nobody thought to ask for it. A leftover checkpoint from a different
        # experiment at the same --out is rejected by the fingerprint check, not resumed.
        ckpt = Path(args.ckpt) if args.ckpt else Path(args.out).with_suffix(".ckpt")
        assert ckpt != Path(args.out), f"--ckpt and --out must differ, both are {ckpt}"
        if ckpt.exists():
            start_step, history = load_checkpoint(
                ckpt, split=split, optimizer=opt, fingerprint=fingerprint,
                defaults={**training_fingerprint({k: p.get_default(k) for k in vars(args)}), "world_size": 1})
            assert 0 <= start_step <= args.steps, f"checkpoint step {start_step} outside [0, {args.steps}]"
            assert len(history) == start_step, f"{len(history)} logged steps for checkpoint step {start_step}"
            if main_rank:
                print(f"[resume] {ckpt}: continuing at step {start_step}/{args.steps}", flush=True)
        result["resumed_from_step"] = start_step
        # Sequences already consumed = steps done * sequences per step; the loader
        # continues the same stream from there.
        assert cfg.train.seqs_per_step % dist_info.world_size == 0, (
            f"sequences per step ({cfg.train.seqs_per_step}) must be divisible by the number of ranks "
            f"({dist_info.world_size})"
        )
        local_seqs = cfg.train.seqs_per_step // dist_info.world_size
        # Rank r reads order[r::R]; after `start_step` steps it has consumed start_step *
        # local_seqs of ITS items, so that is where its stream resumes.
        train_loader = build_dataloader(Path(args.data), "train", args.seq_len, 1,
                                        shuffle=True, seed=args.seed, num_workers=2,
                                        rank=dist_info.rank, world_size=dist_info.world_size,
                                        start_sequence=start_step * local_seqs)
        it = iter(_cycle(train_loader))
        trainer = Trainer(cfg, model, split, loop, opt, it, device=device,
                          empty_cache=args.empty_cache, dist=dist_info)
        for step in range(start_step, args.steps):
            m = trainer.train_step(step)
            history.append(m.as_log())
            # The history is truncated to the checkpointed step on resume by construction:
            # it is saved together with the weights, so a resume replays steps after it.
            if main_rank and ((step + 1) % args.ckpt_every == 0 or step == args.steps - 1):
                save_checkpoint(ckpt, step=step + 1, split=split, optimizer=opt,
                                history=history, fingerprint=fingerprint)
            if main_rank and (step % 5 == 0 or step == args.steps - 1):
                print(f"[train] {m.as_log()}", flush=True)
        result["history"] = history
        result["world_size"] = dist_info.world_size
        if not main_rank:
            # Training is done and every rank holds the same weights. Evaluation involves no
            # collective, so the other ranks leave and the main rank evaluates alone.
            return

    # Shuffle the validation set with a FIXED seed. Deterministic, but it spreads the
    # evaluated sequences across documents instead of walking the first one. Without
    # this, PG-19's val split begins with the King James Bible, which is long enough
    # that 16 sequences of 32768 never leave it - and which the base model has
    # memorised (0.19 nats, 1039 distinct tokens over 7K positions), so the whole
    # evaluation would sit on text that is trivially predictable for every arm.
    val_loader = build_dataloader(Path(args.data), "val", args.seq_len, 1,
                                  shuffle=True, seed=args.seed, num_workers=2)
    probe_batch, probe_info = None, None
    if args.forgetting_probe_tokens > 0:
        probe_batch, probe_info = build_probe_batch(Path(args.data), args.seq_len, args.seed,
                                                    args.eval_sequences, args.forgetting_probe_tokens)
        print(f"[probe] {probe_info}", flush=True)
    t0 = time.perf_counter()
    ev = evaluate(loop, split, val_loader, max_sequences=args.eval_sequences,
                  probe_batch=probe_batch, device=device)
    result["eval"] = {"loss": ev.loss, "num_sequences": ev.num_sequences,
                      "token_nll": ev.token_nll.tolist(),
                      "per_sequence_loss": ev.per_sequence_loss,
                      "forgetting_delta_nll": ev.forgetting_delta_nll,
                      "per_sequence_forgetting": ev.per_sequence_forgetting,
                      "forgetting_probe": probe_info,
                      "seconds": time.perf_counter() - t0}
    if args.eval_ttt_off:
        # SAME trained slow weights, inner loop switched off. This is the only comparison
        # that isolates test-time training itself: an inner_lr=0 *training* run learns a
        # DIFFERENT (non-meta-learned) LoRA, so it measures the whole system against plain
        # fine-tuning, not the contribution of TTT at inference.
        off_cfg = Config(model=cfg.model, inner=replace(cfg.inner, lr_rms=0.0),
                         outer=cfg.outer, train=cfg.train)
        off_loop = TTTInnerLoop(model, off_cfg, build_inner_optimizer(off_cfg.inner))
        off_loader = build_dataloader(Path(args.data), "val", args.seq_len, 1,
                                      shuffle=True, seed=args.seed, num_workers=2)
        t1 = time.perf_counter()
        ev_off = evaluate(off_loop, split, off_loader, max_sequences=args.eval_sequences,
                          device=device)
        result["eval_ttt_off"] = {"loss": ev_off.loss, "num_sequences": ev_off.num_sequences,
                                  "token_nll": ev_off.token_nll.tolist(),
                                  "per_sequence_loss": ev_off.per_sequence_loss,
                                  "seconds": time.perf_counter() - t1}
        print(f"[eval] arm={args.arm} TTT-OFF loss={ev_off.loss:.4f} "
              f"delta_from_ttt={ev_off.loss - ev.loss:+.4f}", flush=True)

    if torch.cuda.is_available():
        result["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"[eval] arm={args.arm} loss={ev.loss:.4f} n={ev.num_sequences} "
          f"peak={result.get('peak_gib', 0):.1f}GiB", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Atomic, like the checkpoint: a job killed mid-write must not leave a truncated file
    # that a later link of the chain would then fail to parse.
    tmp = Path(args.out).with_name(Path(args.out).name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, args.out)
    print(f"[run] wrote {args.out}", flush=True)


def _cycle(loader):
    while True:
        yield from loader


if __name__ == "__main__":
    main()
