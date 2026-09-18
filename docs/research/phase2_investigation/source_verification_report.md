## Source Verification Report (Phase 2)
**Date**: 2026-09-16. **Method**: Tier 1 existence check for every arXiv ID by fetching `https://arxiv.org/abs/<id>` and reading `citation_title` / `citation_author` / `citation_date` meta tags (arXiv API returned HTTP 406 from this network, so the abs-page path was used); Tier 1 for repos/blogs by HTTP status. Semantic Scholar not queried (`[S2-API-UNAVAILABLE]` — not attempted from this environment). Field velocity: rapid (AI/ML) → sources older than 3 years accepted only as foundational.

### Overall Assessment
**Sources reviewed**: 29 arXiv + 14 web/repo. **Verified**: 42 | **Flagged**: 1 (wrong ID in user-supplied list, corrected) | **Rejected**: 0.

### arXiv existence check (all VERIFIED unless noted)
| arXiv | Title (as resolved) | First author | Date | Level* | Note |
|---|---|---|---|---|---|
| 2512.23675 | End-to-End Test-Time Training for Long Context | Tandon | 2025-12 | VI (preprint, primary) | ground truth for this study |
| 2407.04620 | Learning to (Learn at Test Time): RNNs with Expressive Hidden States | Sun | 2024-07 | VI | |
| 2505.23884 | Test-Time Training Done Right (LaCT) | Zhang, T. | 2025-05 | VI | code read by agent |
| 2502.16982 | Muon is Scalable for LLM Training | Liu, J. | 2025-02 | VI | |
| 2501.00663 | Titans: Learning to Memorize at Test Time | Behrouz | 2024-12 | VI | |
| 2505.23735 | ATLAS: Learning to Optimally Memorize the Context at Test Time | Behrouz | 2025-05 | VI | |
| 2512.24695 | Nested Learning: The Illusion of Deep Learning Architectures | Behrouz | 2025-12 | VI (NeurIPS 2025 per e2e refs) | |
| 2608.21308 | Rethinking Expressivity and Efficiency in Test-Time Training (E²-TTT) | Zhong | 2026-08 | VI | |
| 2606.21803 | Test-Time Training with Next-Token Prediction (TTT-NTP) | Ouyang | 2026-06 | VI | |
| 2605.28053 | RW-TTT: Batched Serving for Request-Owned Test-Time Training State | Yang, J. | 2026-05 | VI | user-supplied |
| **2505.17895** | **DataRater: Meta-Learned Dataset Curation** | Calian | 2025-05 | — | **FLAG: user-supplied list attributed this ID to MixFlow-MG; wrong paper. Excluded from corpus.** |
| **2505.00793** | Scalable Meta-Learning via Mixed-Mode Differentiation (MixFlow-MG) | Kemaev | 2025-05 | VI (ICML 2025) | correct ID, substituted |
| 1810.10180 | Understanding and correcting pathologies in the training of learned optimizers | Metz | 2018 | VI (ICML 2019) | foundational |
| 1606.04474 | Learning to learn by gradient descent by gradient descent | Andrychowicz | 2016 | VI (NeurIPS 2016) | foundational |
| 1810.09502 | How to train your MAML | Antoniou | 2018 | VI (ICLR 2019) | foundational |
| 1707.09835 | Meta-SGD | Li, Z. | 2017 | VI | foundational |
| 1909.04630 | Meta-Learning with Implicit Gradients | Rajeswaran | 2019 | VI (NeurIPS 2019) | foundational |
| 2106.09685 | LoRA: Low-Rank Adaptation of Large Language Models | Hu | 2021 | VI (ICLR 2022) | foundational |
| 2312.03732 | A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA) | Kalajdzievski | 2023 | VI | |
| 2402.12354 | LoRA+ | Hayou | 2024 | VI (ICML 2024) | |
| 2402.09353 | DoRA | Liu, S.-Y. | 2024 | VI (ICML 2024) | |
| 2405.09673 | LoRA Learns Less and Forgets Less | Biderman | 2024 | VI (TMLR) | |
| 2507.06415 | PERK: Long-Context Reasoning as Test-Time Learning | Chen, Z. | 2025-07 | VI (ICLR 2026 per agent) | closest precedent |
| 2601.04181 | Lightweight Test-Time Adaptation for EMG-Based Gesture Recognition | Touko | 2026-01 | VI | peripheral (non-LLM) |
| 2603.03524 | Test-Time Meta-Adaptation with Self-Synthesis | Kaya | 2026-03 | VI | |
| 2605.09932 | FocuSFT: Bilevel Optimization for Dilution-Aware Long-Context Fine-Tuning | Pei | 2026-05 | VI | reverse configuration |
| 2410.22264 | Provable Meta-Learning with Low-Rank Adaptations | Block | 2024-10 | VI | |
| 2506.10943 | Self-Adapting Language Models (SEAL) | Zweiger | 2025-06 | VI | |
| 2403.09113 | AutoLoRA | Zhang, R. | 2024-03 | VI | peripheral |

*Level: all are ML preprints/conference papers (single empirical studies) → Level VI on the generic hierarchy; discipline-relative grade A for peer-reviewed venues (ICLR/ICML/NeurIPS/TMLR), B for preprints. No predatory venues. No COI beyond authors evaluating their own methods (intellectual COI, moderate, universal in the field — flagged once here for all).

### Web / repository sources (HTTP 200 on 2026-09-16)
github.com/test-time-training/e2e (commit a4fc478, read locally) · github.com/microsoft/LoRA · gist.github.com/sayakpaul/9bae12402eddd53a79ee1f64b659b07b · thinkingmachines.ai/blog/lora/ (industry blog, Grade B) · github.com/a1600012888/LaCT · kellerjordan.github.io/posts/muon/ (blog, Grade B) · github.com/KellerJordan/Muon · github.com/facebookresearch/higher · github.com/zeyun-zhong/E2-TTT · github.com/doem97/metalora · tridao.me/blog/2026/gram-newton-schulz/ (blog, Grade B) · github.com/shirleyzhu233/PyTorch-MAML · pytorch.org/blog/activation-checkpointing-techniques/ (vendor blog, Grade B) · docs.axolotl.ai/docs/gradient_checkpointing.html (vendor docs, Grade B).

### Flagged Sources (Detail)
#### arXiv:2505.17895 (user-supplied as "MixFlow-MG")
- **Issue**: ID resolves to DataRater (Calian et al.), a different DeepMind meta-learning paper. Likely a copy error in the user's source list.
- **Severity**: Medium (would have propagated a wrong citation). **Recommendation**: Exclude; cite arXiv:2505.00793 instead. Memory-agent brief corrected mid-run.

### Verification Limitations
- Existence and title/author/date verified; claims *inside* each paper were verified by the search agents against fetched HTML/PDF text (PERK, rsLoRA re-verified from extracted PDF after one fabricated summary was discarded) but not independently re-read by the orchestrator. Numerical claims in Phase 3 are therefore attributed to the agent reports and cited to the source, not asserted as independently checked.
- Second-batch verification for the memory/checkpointing agent's references is appended below when that report arrives.

### Batch 2 (memory/checkpointing search layer) — all VERIFIED 2026-09-16
| arXiv | Title (as resolved) | First author | Date | Level |
|---|---|---|---|---|
| 2604.06169 | In-Place Test-Time Training | Feng, G. | 2026-04 | VI (ICLR 2026 per agent) |
| 2603.13875 | GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent | Kuratov | 2026-03 | VI |
| 2607.06988 | WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time | Feng, Y. | 2026-07 | VI (peripheral, non-LLM) |
| 1502.03492 | Gradient-based Hyperparameter Optimization through Reversible Learning | Maclaurin | 2015 | VI (ICML 2015, foundational) |
Repos/docs HTTP 200: github.com/ByteDance-Seed/In-Place-TTT · github.com/yancyou/TTT-NTP · github.com/JarvisPei/FocuSFT · github.com/yurakuratov/gradmem · github.com/test-time-training/ttt-lm-jax · github.com/pytorch/pytorch/issues/91469 · docs.jax.dev memory-spaces + gradient-checkpointing pages · proceedings.mlr.press/v267/kemaev25a.html (MixFlow-MG, ICML 2025 → Grade A).
**Final tally**: 33 arXiv + 23 web/repo sources verified; 1 excluded (2505.17895 wrong-ID); 0 fabricated. Source base quality: Strong (all venues legitimate; 2026 preprints are unrefereed and graded B).
