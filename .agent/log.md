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
