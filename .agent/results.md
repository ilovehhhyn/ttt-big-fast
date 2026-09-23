# Results: the headline numbers and where each one lives

Every number below is copied from `docs/results/results.md` (R) or from a Della result file,
with the section that holds its context and interval. Loss is nats per token. "Books positive"
is the count of documents whose paired difference favours the first condition. Nothing here is
estimated; a pending number says so. Updated 2026-09-23.

## Baselines with nothing trained (PG-19, T = 32768, 32 sequences from 22 books)

| model | loss | R section |
|---|---|---|
| full attention, no TTT | 2.3092 | "The full-attention diagnostic" |
| window 8192, no TTT (arm A) | 3.7119 | "TTT alone (arm B against arm A)" |
| window 8192, TTT at 4e-6 (arm B) | 3.5691 | same; +0.1405 per book [+0.1279, +0.1531], 22/22 |
| window 1024, no TTT | 4.9895 | "Nothing trained, k = 1024" |
| window 1024, TTT at 4e-6 | 4.5232 | same; best rate 7e-6 gives 4.5199 |
| 8K context, window 8192 (T/k = 1): A / B | 2.3067 / 2.3078 | "The matched T/k = 1 row"; TTT slightly harmful, -0.0016 per book |
| 128K context, window 8192: A / B (16 books) | 4.2900 / 3.9290 | "PG-19 at 128K, nothing trained"; +0.3610 per book, 16/16 |
| SlimPajama, window 1024, 96 sequences: A / B | 4.3094 / 3.8821 | "SlimPajama at k = 1024: nothing trained" |

Position pattern at window 8192 (R, "TTT alone"): inside the window TTT changes nothing
(-0.0013); beyond it the gain grows with position (+0.1132, +0.2041, +0.2550 by 8192-token band).
The same weights with full attention are flat at about 2.3092 in every band, so the un-tuned
windowed model is broken beyond the window and TTT mostly repairs it.

## What old context is worth to a healthy model (the ceiling for any memory)

`scripts/context_value.py`, tightest band (restart keeps at least 7/8 of a window of recent
tokens), per document. R, "What context beyond the window is worth" and "Where out-of-window
context is worth most".

| corpus, window S | value | 95% CI |
|---|---|---|
| PG-19, 8192 | +0.0208 | [+0.0139, +0.0277] |
| PG-19, 2048 | +0.0667 | [+0.0544, +0.0790] |
| PG-19, 1024 | +0.0992 | [+0.0839, +0.1145] |
| PG-19 at 128K, 8192 (two pieces of 6 books) | +0.0323, +0.0385 | [-0.0000, +0.0647], [+0.0197, +0.0573] |
| SlimPajama arXiv, 8192 | +0.0814 | [+0.0209, +0.1419] |
| SlimPajama arXiv, 1024 | +0.2548 | [+0.1804, +0.3291] |
| SlimPajama GitHub, 1024 | +0.2133 | [+0.1439, +0.2826] |
| SlimPajama Book, 8192 | +0.0325 | [+0.0177, +0.0472] |

## The 2x2 behind H1 (cells: TTT on / TTT off; interaction per document)

| setting | trained through the inner loop | plain fine-tune | interaction | 95% CI | positive | R section |
|---|---|---|---|---|---|---|
| window 8192, 10 steps | 2.6688 / 2.6888 | 2.6944 / 2.7124 | +0.0015 | [+0.0005, +0.0026] | 17/22 | "The 2x2 behind H1" |
| window 1024, 10 steps | 2.9440 / 3.0041 | 3.0212 / 3.0682 | +0.0126 | [+0.0098, +0.0153] | 22/22 | "Arm C at k = 1024, 10 steps" |
| window 1024, 40 steps | 2.6777 / 2.7086 | 2.6979 / 2.7196 | +0.0083 | [+0.0062, +0.0104] | 22/22 | "Arm C at k = 1024, 40 steps" |
| SlimPajama, window 1024, 40 steps, 96 sequences | 2.2789 / 2.3193 | 2.2920 / 2.3252 | +0.0074 | [+0.0056, +0.0091] | 85/88 | "SlimPajama at k = 1024" |

From 10 to 40 steps at window 1024 every effect shrank; the interaction least (-34%), the
training effect with TTT off most (-85%). Across the 88 SlimPajama documents the interaction
follows the ceiling (coefficient +0.040 [+0.028, +0.053]) and not the damage (+0.009
[-0.004, +0.021]); the coefficient survives adding the loss level (+0.032) and domain indicators
(+0.032). The healthy level at window 1024 is about 2.39, so all cells sit above it.

## Memory measured directly: the recall test

PG-19, window 1024, passage of 1024 tokens planted at 2048, repeated at 20480 (gap 17408 >
16 x 1023), first 32 tokens of the repeat not scored, 32 pairs from 20 books. Recall = loss on
the repeat without the first copy minus with it. Without TTT recall is exactly 0 in every run.
R, "Recall test", "More recall runs", "Recall against the inner learning rate", "Two write
rules that equalize the update".

| weights | inner rule, rate | recall | 95% CI | ordinary loss |
|---|---|---|---|---|
| un-tuned | normalized SGD 4e-6 | +0.0717 | [+0.0651, +0.0783] | 4.5232 |
| un-tuned | normalized SGD 1.4e-5 (best for memory) | +0.1563 | [+0.1263, +0.1862] | 4.6035 |
| un-tuned | preconditioned SGD 4e-5 | +0.4858 | [+0.4368, +0.5348] | 4.5517 |
| un-tuned | Muon 1.2e-4 | +0.6865 | [+0.6488, +0.7242] | 4.3290 |
| un-tuned | full attention, no TTT (ceiling) | +2.6920 | [+2.4448, +2.9393] | 2.3092 |
| 40 steps through the inner loop, at 4e-6 | normalized SGD 4e-6 | +0.1054 | [+0.1010, +0.1097] | 2.6777 |
| 40 steps plain fine-tune | normalized SGD 4e-6 | +0.1018 | [+0.0976, +0.1060] | 2.6979 |
| 40 steps at 4e-6, tested at | normalized SGD 2e-5 | +0.3880 | [+0.3105, +0.4655] | 2.8318 |
| 40 steps TRAINED at 1e-5 | normalized SGD 1e-5 | +0.2931 | [+0.2770, +0.3091] | 2.6820 |
| 40 steps at 4e-6, tested at | preconditioned SGD 4e-5 | +0.7025 | [+0.5986, +0.8064] | 2.7648 |
| 40 steps at 4e-6, tested at | Muon 4e-5 | +0.4183 | [+0.3918, +0.4448] | 2.6636 |
| 40 steps at 4e-6, tested at | Muon 1.2e-4 | +1.0241 | [+0.9555, +1.0927] | 2.6895 |
| 40 steps at 4e-6, tested at | Muon 2.4e-4 | +1.5005 | [+1.3894, +1.6117] | 2.7937 |
| 40 steps TRAINED at 2e-5 | normalized SGD 2e-5 | +0.5201 | [+0.4733, +0.5669] | 2.7243 |

Other recall facts: trained through the inner loop minus plain fine-tune +0.0036
[+0.0018, +0.0053], 18/20 books; recall is flat along the passage (+0.0707, +0.0722, +0.0788,
+0.0731 by quarter); 2 / 4 / 8 fast blocks give +0.0413 / +0.0717 / +0.1021; with the copies
4096 apart (inside attention's reach) recall without TTT is -0.0007; window 8192 un-tuned
+0.0283 and 10-step weights +0.0772 (16 pairs); SlimPajama un-tuned +0.1076, 40-step +0.0971,
full attention +2.2611. Both preconditioned and Muon are unstable at 1e-4 and 5e-5 respectively
for normalized SGD; Muon is still stable at 2.4e-4.

The shared key directions (`scripts/key_basis.py`, 65,536 training tokens): in each fast block
the strongest direction carries 32 to 60% of the key energy and 64 directions carry 55 to 83%
(un-tuned); 47 to 73% on the 40-step weights.

## Longer training at the reference window (window 8192, normalized SGD 4e-6; R, "2026-09-23")

| run | steps x sequences | loss, TTT on | TTT off | on - off per book | 95% CI |
|---|---|---|---|---|---|
| `C32k_t2` (through the inner loop) | 20 x 4 | 2.5958 | not evaluated | against the plain control: +0.0186 | [+0.0146, +0.0226], 21/22 |
| `C32k_ctl20` (plain fine-tune) | 20 x 4 | 2.6139 | | | |
| `C32k_t1` (`truncate_bptt=1`) | 20 x 4 | 2.5966 | | against `C32k_t2`: -0.0004 | [-0.0014, +0.0006], 11/22 |
| `C32k_bs32` | 20 x 32 | 2.5664 | 2.5777 | +0.0091 | [+0.0047, +0.0134], 22/22 |
| `C32k_s60` | 60 x 4 | 2.5146 | 2.5294 | +0.0125 | [+0.0082, +0.0167], 22/22 |
| `C32k_s150` | 150 x 4 | 2.4664 | 2.4756 | +0.0067 | [+0.0025, +0.0110], 21/22 |

`C32k_bs8` and `bs16` (20 x 8, 20 x 16): 2.5799 / 2.5943 and 2.5738 / 2.5909 (on / off).

## Other measured facts

| fact | number | R or F section |
|---|---|---|
| AdamW as the inner rule, window 8192, un-tuned, best rate 2e-5 | 3.6175 (normalized SGD: 3.5691) | R, "AdamW: the full curve" |
| arm D (all weights slow), 10 steps, `truncate_bptt=1` | outer lr 4e-4 diverges (7.0392); 4e-5 gives 2.7306 | R, "Arm D" |
| arm C 10 steps at window 8192 against its plain control | 2.6690 against 2.7125; the LoRA does not improve in-window loss | R, "Arm C at 32K" |
| same weights, TTT on against off (8-step arm C) | +0.0248 per book [+0.0194, +0.0301], 22/22 | R, "Same weights, inner loop on and off" |
| run-to-run noise, same config, different node | 2.49e-4 in loss | R, "Checkpoint / resume" |
| 2 GPUs against 1 process, same global batch | largest loss difference 3.737e-04; 65.7 against 131.9 s per step | R, "Data parallelism over sequences" |
| step time, window 1024, truncation 4, 4 sequences | 55 s (control 30 s); peak 42.0 GiB | R, "A smaller window affords a longer truncation window" |
| step time, window 8192, truncation 2 | 131.9 s; peak 68 GiB | F section 13 |
| Muon evaluation pass, A100 | 20 s per sequence against 3.4 | R, "Two write rules" |

## Corpora (Llama-3 tokenizer, BOS between documents)

| corpus | training tokens / documents | validation tokens / documents |
|---|---|---|
| `pg19_32k` (books >= 32,769 tokens, every 40th held out) | 300,119,276 / 2,487 | 8,708,086 / 64 |
| `pg19_32k_full` (same rule over the whole stream; same validation bytes) | 900,080,292 / 7,626 | 8,708,086 / 64 |
| `pg19_128k` (books >= 131,073 tokens, every 25th held out) | 1,466,573,365 / 6,505 | 66,629,305 / 272 |
| `slimpajama_32k` (`DKYoon/SlimPajama-6B`, documents >= 32,769 tokens, every 10th held out, source label kept) | 476,513,860 / 6,176 | 53,238,246 / 687 |

## Pending numbers

Muon meta-training memory and speed (jobs 14330212, 14330213); the 20-step 2x2 at window 8192
(`login_cells_k8192_s20.sh`); the 60- and 150-step plain controls (14330258, 14330259);
`C32k_bs32s60` (14169729, running); the matched-budget chains in `plan.md`.

