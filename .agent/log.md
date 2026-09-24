# Log

One entry per day. Later entries correct earlier ones in the open. Details are in
`docs/results/results.md` (R) and `docs/research/FINDINGS.md` (F).

- 2026-09-16. Scoping. Read the TTT-E2E paper and code; wrote the research question brief and
  the first gap analysis (`docs/research/phase1_scoping/`).
- 2026-09-17. Plan written for a JAX fork of `e2e` (`docs/superpowers/plans/2026-09-17-*.md`).
  Decided: normalized SGD and AdamW both required as inner rules; free data only.
- 2026-09-18. Switched to PyTorch for modular code (`2026-09-18-pytorch-architecture.md`).
  Model, data, inner loop, optimizers, evaluation and tests written. First 8K runs on DCLM.
  Found that the inner rate must be in the unit 1/sqrt(n_fast): 1e-3 diverges.
- 2026-09-19. Made 32K fit on one 80 GiB GPU: per-window backward for truncated BPTT, one
  checkpoint region per chunk, checkpointed prefix segments, `inference=True` for evaluation.
  Step time 455.8 to 131.9 s. Four memory diagnoses were wrong before tracing (F section 13).
  First arm C at 32K: 2.6690 against a plain fine-tuning control 2.7125. Exact resume validated.
- 2026-09-20. The window damage: full attention 2.3092 against 3.7119 at window 8192; old
  context worth +0.0208 at 8192, +0.0992 at 1024. The 2x2: interaction +0.0015 at window 8192.
  Arm D's slow set had been empty (bug fixed); arm D diverges at 4e-4. Check-in report written
  and audited (`docs/research/checkin-2026-09-20.md`). Round-2 direction from Helen: window
  1024, longer context or a task with long-range dependence, SlimPajama, match the reference
  budget.
- 2026-09-21. Data parallelism by hand validated (2 GPUs equal 1 process). Corpus extended to
  900M tokens; matched-budget chains queued at both windows. Window 1024: interaction +0.0126
  (10 steps), +0.0083 (40 steps). SlimPajama and 128K corpora built; per-domain ceilings; 128K
  A 4.2900, B 3.9290. Recall test built and run: 4% of full attention. Five sub-agents
  diagnosed the weak write. Inner-rate sweeps; Muon and the preconditioned rule measured on
  Llama: Muon 1.2e-4 reaches 38% of full attention. Muon meta-training validation submitted.
  Helen asked for plain textbook English and short messages (recorded in memory).
- 2026-09-22. Summaries and the mentor update drafted. Decision: no KV cache in the method.
- 2026-09-23. The window-8192 ladder queued on 2026-09-20 ran: 20 steps at 4 to 32
  sequences, 60 and 150 steps (R, "2026-09-23"). What TTT adds on the same weights shrinks with
  training, to +0.0067 at 150 steps. Muon at 2.4e-4 recalls 56% of full attention. Weights
  trained at 2e-5 recall +0.5201. The Muon meta-training validation had been submitted with
  `--steps 3`, which the warmup check rejects (the checklist says 5); resubmitted. Plain
  controls for the 60- and 150-step runs and the 20-step 2x2 launched. This handoff folder
  written. Helen's PLI account (`pli_x`, H100 nodes) verified; the four matched-budget
  chains moved there (jobs 14330761 to 14330768), the `arora` copies cancelled.
  The 2x2 at window 8192 and 20 steps: interaction +0.0058, 22/22 books (10 steps: +0.0015).
  Meta-training through Muon fits (286 s per step, 60.4 GiB); the 40-step run submitted as a
  chain of short-queue links (14333213 to 14333217). 60-step plain control: 2.5255.
  Evening: the 60-step 2x2 at window 8192 landed: interaction +0.0077 [+0.0058, +0.0096],
  21/22 books; with TTT off the plain fine-tune is now slightly better (-0.0024). New agent
  took over at 18:30; suite passed locally (237 tests); Muon 40-step link 1 running at 301 s
  per step on its A100 (286 in the validation job).
  Later: Helen adopted five LaCT ideas (plan.md, "Planned after that") and, after seeing the
  1.9% bf16 difference, the bf16 Newton-Schulz change too. Built and tested locally, in
  order: `--ns-dtype bfloat16`, `--weight-norm row_reset`, `--token-rates`, and arm F
  (`--prime-intermediate`, `fast_init_trained`); 262 tests pass. Nothing measured on Llama:
  the Della SSH session lapsed at about 19:30 and did not come back during the session.
  choobi updated `.agent/README.md` (it cannot see `plan.md`); the rest by hand.
- 2026-09-24. 00:59 Della back. The Muon 40-step run had finished: loss 2.6762 (TTT off
  2.7380), recall +1.0006, against +1.0241 for the normalized-SGD-trained weights under the same
  write (paired -0.0235, 2/20). Loss prediction met, recall prediction failed. The login node
  killed the plain control's Muon evaluation at 13 minutes; it moved to `gpu-test`
  (`recall_muon.sbatch` now also scores TTT off). Memory probe at chunk 2048: 49.73 GiB. The
  trainer left 0.8 GiB of unused gradient on the fast weights every run; fixed. Queued: the
  control's Muon cell, the bf16 and row-reset variants on the 40-step weights, the 150-step
  control (gpu-test nodes unavailable overnight).
  02:00 to 04:30: 2x2 under Muon, interaction +0.0492 (mostly dependence); bf16 Newton-Schulz
  same numbers and 2 to 4x faster at evaluation, 3.5x in meta-training: adopted. Row reset:
  nothing at 1.2e-4 or 4.8e-4: dropped. Muon 4.8e-4 recalls 71% of full attention at loss
  3.06. Token rates: no loss change in 40 steps. Arm F: zero gate never opens; gate init flag
  added. Literature read on how the field measures (loss first, needles second). Chain
  through Muon 2.4e-4 submitted.
  05:00 to 06:40: bs32_s60 (60 x 32) equals s150 (150 x 4) at three times the tokens. Arm F
  with gates at 0.1 uses its write (+0.0130) but trails arm C by 0.040. Token-rate recall a
  null. Chain through Muon 2.4e-4 (bf16): loss 2.7452, recall +1.4570; both predictions met;
  the strong write hurts plain weights (-0.0905) and helps weights trained through it. The
  150-step control resubmitted as a 2-GPU chain (link 1 reached step 107, link 2 resumed).
  A stale inline launcher blocked the login GPU queue for an hour; killed; login scripts now
  wait on GPU processes and live in files.
