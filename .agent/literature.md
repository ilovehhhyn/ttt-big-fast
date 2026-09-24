# Literature: where the design and the fixes came from

This file lists the papers and code the project drew on, what was taken from each, and how far
each claim was checked. The check column is honest about who verified what: "lead, in source"
means the project lead opened the paper or code during this project; "sub-agent, 2026-09-21"
means one of five research sub-agents reported reading it that day and the lead did not
re-check the numbers; "from memory" means neither. Do not cite an unchecked number in a paper
without opening the source. Bibliography entries for the first group are in
`docs/preprint/refs.bib`.

## The reference method

| source | what we took | check |
|---|---|---|
| TTT-E2E, arXiv 2512.23675, "End-to-End Test-Time Training for Long Context"; code `github.com/test-time-training/e2e` (clone on Della, commit a4fc478) | the whole protocol: sliding window k = 8192, chunk b = 1024, one inner step per chunk on next-token loss, loss before the update (Eq. 6), fast MLPs in the last quarter of blocks, second-order outer loop, extension training at 32K on books for 725 steps x 32 sequences, outer lr 4e-4, `ilr_init: 1`, inner rule `clip_by_global_norm(1)` + `sgd(lr=1)` (step norm min(‖g‖, 1)), loss in nats | lead, in source: the experiment YAMLs in `configs/experiment/` (the dataclass defaults are NOT what they ran), `ttt/model/loss.py`, `ttt/optimizers.py` |
| same paper, Table 2 | needle-in-a-haystack at 32K: TTT-E2E 0.24, sliding window alone 0.26, full attention 1.00; the authors write that their mechanism "leaves out seemingly irrelevant details, such as the target string" | lead, in source (arXiv HTML, 2026-09-22) |

Our deliberate departures: fast weights are the full pretrained MLPs (201M) instead of small
extra MLPs; slow weights are a LoRA + norms + learned step sizes instead of every parameter;
the inner step is strictly normalized every chunk; `ilr_init` is 0.1; the meta-gradient is
truncated to 2 to 4 chunks (memory); 4 sequences per step in all short runs.

## Design pieces

| source | what we took | check |
|---|---|---|
| LoRA, Hu et al. 2022 | low-rank adapters as the slow set | from memory; in `refs.bib` |
| rsLoRA, arXiv 2312.03732 (Kalajdzievski) | scale alpha / sqrt(r), so the rank can be swept without retuning the outer rate (`ttt/model/lora.py`) | the code docstring cites its Theorem 3.2; the paper was not re-opened during this project |
| MAML (Finn et al. 2017), Meta-SGD (Li et al. 2017) | second-order meta-learning of an initialization and of per-parameter step sizes (our `inner_lr_log`) | from memory; in `refs.bib` |
| Muon (Keller Jordan, 2024, blog post) | the 5-step Newton-Schulz orthogonalization with coefficients (3.4445, -4.7750, 2.0315) in `ttt/optim/inner.py` | lead, coefficients copied into the code on 2026-09-18; blog URL not re-checked |
| PERK, arXiv 2507.06415 | precedent for a truncated (biased) meta-gradient in test-time learning | sub-agent, 2026-09-21 |
| Dynamic evaluation, Krause et al., arXiv 1709.07432 | the "arm B" idea: gradient steps at test time with no meta-learning; steps every few tokens, decay toward the original weights | sub-agent, 2026-09-21 |

## Why the write was weak, and the fix (2026-09-21)

| source | what we took | check |
|---|---|---|
| LaCT, arXiv 2505.23884, "Test-Time Training Done Right" | Muon as the inner update rule; "Muon consistently outperforming other optimizers" (Figure 7 caption). The sub-agent also reported Table 8: at 760M on S-NIAH-1 at 32K a plain gradient step 14.8, momentum 84.8, Muon 92.4 | title and the Figure 7 sentence: lead, in source (2026-09-22); Table 8 numbers: sub-agent only |
| LaCT, same paper, read in full on 2026-09-23 | the five items in `plan.md`, "Planned after that": chunk 2048 (App. C.2), the Muon break-even chunk (5/3) hd (App. A, Eq. 17 and 18), RMSNorm plus zero-initialised gate on the fast output (Alg. 2, App. C.3), per-token learning rates softplus(Linear(x) + bias) (Eq. 4, Alg. 1 and 2), L2 row normalization after each update with no weight decay (Alg. 1 and 3, Sec. 3.2). Not stated in the paper: the Newton-Schulz precision, a fixed momentum value, outer weight decay and clipping for the LM runs | sub-agent, 2026-09-23, all 32 pages of the PDF; figure curves not read; the lead has not re-opened these sections |
| MEMIT, arXiv 2210.07229, Eq. 14 | writes of the form R K^T (C0 + K K^T)^-1: the key covariance C0 = E[k k^T] whitens the write. Our `preconditioned_sgd` removes the top eigenvectors of that covariance instead of inverting it | sub-agent, 2026-09-21 (equation checked by the agent) |
| ROME, Meng et al. 2022 | factual associations live in middle-layer MLPs. The sub-agent's replica found the opposite for our kind of write: the LAST blocks read back best (0.91 of self-recall against 0.65 for early blocks) | sub-agent; arXiv id not recorded |
| TTT-NTP, arXiv 2606.21803 | "whitening is decisive": a raw one-shot Hebbian write scores 12.6 on RULER, the whitened write 59.7 (Table 3) | sub-agent, 2026-09-21 |
| In-Place TTT, arXiv 2604.06169; TTCD, arXiv 2608.01672 | Hebbian writes into the MLP down-projection; RULER at 8K from 9.91 to 26.80 against full attention 38.09; TTCD's write target comes from a longer-window copy of the model | sub-agent, 2026-09-21 |
| DeltaNet and Gated DeltaNet, arXiv 2412.06464 | the delta rule (erase, then write) "helps memorization"; "decay hurts memory retention" (section 3.2) | sub-agent, 2026-09-21 |
| Titans, arXiv 2501.00663; ATLAS, arXiv 2505.23735 | momentum, weight decay and deep memory; ATLAS fits a window of past tokens with Muon | sub-agent, 2026-09-21 |
| FwPKM, arXiv 2601.00671 | one gradient pass stores little (needle under 10%), repeated passes much more (over 70%) | sub-agent, 2026-09-21 |
| Physics of language models, arXiv 2404.05405; arXiv 2505.24832 | about 2 to 3.6 bits per parameter of capacity, reached only after many exposures: at 32K to 128K tokens the limit is the write rule, not capacity | sub-agent, 2026-09-21 |

## Limits of a fixed-size memory

| source | what it says | check |
|---|---|---|
| "Repeat After Me" (Jelassi et al.), arXiv 2402.01032, Theorem 2.7 | a state with fewer than L log2(D) - 1 bits copies a random L-token string over D symbols with error above 1/2 | sub-agent, 2026-09-21 |
| Zoology / Based (Arora, Eyuboglu et al.), arXiv 2402.18668, 2402.18510 | multi-query associative recall needs state that grows with the sequence; one attention layer or retrieval closes the gap | sub-agent, 2026-09-21 |
| the sub-agent's synthesis | no fixed-size memory in the literature matches a well-trained full-attention model beyond 16K; the best case reported is 48 against 91 at 32K | sub-agent's reading; treat as a claim to re-check |

## Cache hybrids (read, not adopted)

| source | what it says | check |
|---|---|---|
| Jamba, arXiv 2403.19887; Griffin, arXiv 2402.19427; NSA, arXiv 2502.11089; Samba, arXiv 2406.07522 | a few attention layers or a sparse attention over stored tokens restore needle retrieval; NSA reads 16 blocks of 64 tokens plus a 512-token window | sub-agent, 2026-09-21 |
| StreamingLLM, arXiv 2309.17453; H2O, arXiv 2306.14048; Quest, arXiv 2406.10774; KIVI, arXiv 2402.02750; DuoAttention, arXiv 2410.10819 | attention sinks, eviction by attention mass, block retrieval, 2 to 4 bit caches | sub-agent, abstracts only |

Decision (Helen, 2026-09-22): no cache beyond the window in the method. Recall would then come
from the cache and not from the fast weights, which is a different claim. The attention-sink
observation is still worth one evaluation-only test, because it may explain the window damage.

## Sub-agent reports

The five reports of 2026-09-21 (update rule, KV cache, what is written and read, slow weights,
literature) are not in the repository; their scratch scripts were under the session's
scratchpad and may be gone. Their measured claims were on SmolLM2-135M or toy models and are
summarized in `docs/results/results.md`, "Why the write is weak". Nothing from them is a
Llama measurement.
