# Handoff: big fast weights, small slow weights

This folder is the entry point for an agent or a person taking over the `ttt-big-fast` project.
It says what the project tests, where every document and result lives, what the numbers say
today, what is running, and what to do next. It was written on 2026-09-23 from the repository,
the result files on Della and the project lead's memory notes. Read this file first, then
`plan.md`, then `results.md`.

| file | what it holds |
|---|---|
| `README.md` | this map, the thesis, the state in ten lines, the rules of the project |
| `plan.md` | hypotheses, arms, protocol, decision rules, what is queued and planned, predictions |
| `results.md` | every headline number with its source section in `docs/results/results.md` |
| `literature.md` | the papers the design and the fixes came from, and how far each claim was checked |
| `operations.md` | Della, scripts, how a run is launched, and the mistakes that already cost a job |
| `log.md` | dated log of what happened, one entry per day |
| `HANDOFF_PROMPT.md` | the message to paste into a fresh agent's first turn: reading order, rules, state, next steps, past mistakes |

## The project in one paragraph

TTT-E2E (arXiv 2512.23675) trains a language model with sliding-window attention whose MLPs in
the last quarter of the blocks are "fast weights": at test time they take one gradient step per
1024-token chunk on that chunk's next-token loss, so the model keeps learning from the text it
reads. Their fast weights are small extra MLPs and every other parameter is meta-learned. This
project inverts that: the fast weights are the FULL pretrained MLP matrices of the last 4 of
Llama-3.2-1B's 16 blocks (201M parameters), and the slow, meta-learned weights are small: a
rank-64 LoRA on the attention and MLP projections, the norm gains, and one learned step size per
fast tensor (45M parameters). The thesis (H1) is that this small slow set, trained through the
inner loop, captures most of what test-time training can give. The code is PyTorch
(`ttt/`), the cluster is Princeton Della, all data is free.

## The state in ten lines (2026-09-23)

1. The sliding window breaks the pretrained model: loss 2.3092 with full attention, 3.7119
   at window 8192, 4.9895 at window 1024 (32K tokens, PG-19). Most of what TTT gains on the
   un-tuned model repairs that damage; old context is worth only +0.0208 nats at window 8192.
2. The thesis is weakly supported. Training the LoRA through the inner loop makes TTT more
   useful in 22/22 books, but by +0.0083 nats (0.3% of the loss) after 40 steps.
3. A recall test (plant a passage, repeat it past attention's reach) measures memory directly.
   With the original write rule TTT stores 4% of what full attention recalls.
4. The write is weak because the gradient is dominated by a few input directions shared by all
   tokens, which caps the step size. Two write rules that spread the update evenly fix this.
5. Muon (all singular values of the gradient set to 1) at a 30x larger step raises recall to
   38% of full attention on the 40-step weights (+1.0241 against +0.1054), at almost the same
   loss (2.6895 against 2.6777).
6. The inner learning rate that is best for loss (4e-6 to 7e-6) is far below the rate that is
   best for memory; training at a larger rate keeps the memory at no cost in loss.
7. We have not matched TTT-E2E's headline: they report parity with full attention at 32K after
   725 steps of 32 sequences. Our runs used 300 to 600x less training. Matched-budget runs
   are queued (about a week's wait).
8. At the reference window (8192), the 60-step 2x2 interaction is +0.0077 [+0.0058, +0.0096]
   across 21/22 books. What TTT adds on the same weights falls from +0.0248 at 8 steps to
   +0.0067 at 150 steps, under the +0.0208 ceiling.
9. Next: finish the running 40-step Muon meta-training run and its control, then test five LaCT
   ideas: a 2048-token chunk arm with Muon's roughly 3400-token cost break-even included in the
   comparison; arm F with RMSNorm on the fast output and a zero-initialized gate; per-token
   meta-learned learning rates; and L2 row normalization without weight decay, first at evaluation
   time with the recall test. The bf16 Newton-Schulz iteration is deferred.
10. The Della SSH session lapses often. Only Helen can sign in; never enter her password.

## Where things live

| what | where |
|---|---|
| code | `ttt/` (model, data, inner loop, optimizers, eval), `scripts/` (experiments), `tests/` |
| canonical results, dated sections | `docs/results/results.md` |
| engineering findings (memory, OOMs, reference cross-check) | `docs/research/FINDINGS.md` |
| the 2026-09-20 check-in report (numbers audited against the docs) | `docs/research/checkin-2026-09-20.md` |
| original plan (JAX) and the PyTorch addendum | `docs/superpowers/plans/2026-09-17-*.md`, `2026-09-18-*.md` |
| scoping and literature phases | `docs/research/phase1_scoping/`, `phase2_investigation/`, `phase3_analysis/` |
| preprint draft (out of date: describes the JAX plan) | `docs/preprint/main.tex`, `refs.bib` |
| house style for code and docs | `docs/agent-style-guide/STYLE_GUIDE.md` (the `mdx` skill) |
| result files and checkpoints (not in git) | Della `/scratch/gpfs/ARORA/hh9077/results/` |
| logs | Della `/scratch/gpfs/ARORA/hh9077/logs/` |
| corpora | Della `/scratch/gpfs/ARORA/hh9077/data/{pg19_32k,pg19_32k_full,pg19_128k,slimpajama_32k}` |
| reference implementation (read only) | Della `/scratch/gpfs/ARORA/hh9077/e2e` |
| GitHub | `ilovehhhyn/ttt-big-fast`, branch `main`; Della mirror at `/scratch/gpfs/ARORA/hh9077/ttt-big-fast` |

## Rules of the project

- Free data and tools only. All cluster work stays under `/scratch/gpfs/ARORA/hh9077`.
- Never enter Helen's password or use a token pasted into chat. When SSH lapses, ask her.
- Every setting that cannot be honoured is a hard error at construction, with the fix named.
- Before any `sbatch`: critique the diff, run the full test suite locally and gate on its exit
  code, run a short validation job that reaches evaluation, then size the long job from a batch
  node's step time (`operations.md`, "Before every sbatch").
- Isolate one factor by holding everything else identical (same weights, switch on and off).
  Cluster paired differences by document before quoting an interval.
- State a prediction before a result comes back, and run the control that could undercut a
  headline number.
- Do not adopt the bf16 Newton-Schulz iteration until its speed gain is measured. On a
  2048 x 8192 matrix it differs from fp32 by 1.9% in relative Frobenius norm.
- Write in plain textbook English. Load the `mdx` skill before touching code or docs.

What this folder does not do: it does not repeat `docs/results/results.md`. Every number here
points at the section that holds its context and its confidence interval.
