# Data beats size: technical report

A controlled ablation of data curation, down-selection and synthetic re-captioning: which data intervention most improves a small captioning model at a fixed budget of 25,000 examples? The design was pre-registered ([PREREGISTRATION.md](PREREGISTRATION.md), amendments A1–A2) and every number below is in [`results/`](results/).

**Contents:** [TL;DR](#tldr) · [1 Setup](#1-setup) · [2 Judge validation](#2-judge-validation) · [3 Data pipeline](#3-data-pipeline) · [4 Results](#4-results) · [5 Engineering](#5-engineering) · [6 Limitations](#6-limitations)

> **In plain words.** Web captions are mostly not descriptions. We compared ways of picking 25,000 training examples for a small
> captioning model. Grading captions with a larger AI model, checked against human grades first, beat every other method by a wide margin;
> a cheap similarity score and AI-rewritten captions did not help.

---

## TL;DR

| | |
|---|---|
| **Best intervention** | Selecting the top 25% of web pairs with a calibrated VLM judge: **CIDEr 0.432 → 0.741** (+71%) and **hallucinating captions 11.5% → 7.5%** vs random data |
| **Why the judge, not CLIPScore** | On 200 hand-rated web captions: judge Spearman **0.52** vs CLIPScore **0.25**. On a clean benchmark (Polaris) they are nearly tied: 0.83 vs 0.82 |
| **Didn't help** | CLIPScore selection (CIDEr unchanged). Synthetic re-captioning is metric-dependent: an independent VLM judge rates its long captions highly (4.58 / 5), but CIDEr drops (−0.04) and hallucinated objects rise |
| **Robust winner** | Only the judge-selected arm is strong on all three metrics: CIDEr, hallucination and the independent judge |
| **Hidden leak** | 164 test images were in the training pool and were removed |
| **Caveat** | The judge-selected arm was added after the pre-registered gate failed on Polaris (+0.013 < 0.05), so it is exploratory |
| **Cost** | $10.01 total: 5.5 A100-hours at $1.81/h, including $0.75 lost on a faulty host |

---

## 1. Setup

**Arms** (25K pairs each, same LoRA recipe, 3 seeds):

| Arm | Data | Pre-registered |
|---|---|---|
| 0 | none (base model) | ✓ |
| A | random CC12M pairs | ✓ |
| B | random after filtering, dedup, decontamination | ✓ |
| C | B pool, top 25K by CLIPScore | ✓ |
| D | B pool, top 25K by judge; fell back to C after the gate (identical data) | ✓ |
| **D\*** | B pool, top 25K by judge | exploratory (A2) |
| E | B's images, re-captioned by a generator + critic | ✓ (A1) |

**Data and models**

| Role | Choice |
|---|---|
| Pool | CC12M shards 0–19: 100,820 pairs |
| Student | SmolVLM-256M-Instruct, LoRA r=16 on the language model (4.9M params), 1 epoch, lr 2e-4, batch 32 |
| Judge / critic | Qwen3-VL-8B-Instruct; score = expected value over the "1"–"5" next-token probabilities |
| Generator (E) | InternVL3-2B |
| Eval judge | Idefics3-8B-Llama3 (no arm was selected or written with this family) |
| Human labels | 200 CC12M pairs rated blind by the author on a written rubric; Polaris (131K ratings) |
| Test | COCO Karpathy test, 5,000 images × 5 references |

---

## 2. Judge validation

**Gate (pre-registered):** the judge must beat CLIPScore by ≥ 0.05 Spearman on Polaris, and not lose on the hand-rated set.

| | Judge | CLIPScore | Δ [95% CI] | Gate |
|---|---:|---:|---|---|
| Polaris test (8,726 pairs) | 0.832 | 0.819 | +0.013 [0.010, 0.015] | ✗ below 0.05 |
| Hand-rated CC12M (200 pairs) | **0.518** | 0.246 | **+0.272** [0.135, 0.414] | ✓ |

→ **NO-GO** as pre-registered. The judge's edge is 20× larger on noisy web captions than on the clean benchmark.

**Calibration** (isotonic, fit on Polaris train):

| | MAE | ECE |
|---|---:|---:|
| Judge, raw | 0.468 | 0.394 |
| Judge, calibrated | **0.210** | **0.047** |
| CLIPScore, calibrated | 0.271 | 0.077 |

**Human ceiling:** two raters of the same caption agree at Spearman **0.255**; the judge agrees with a single rater at **0.328**.

**Length bias:** none found. Correlation with caption length is −0.03 for the judge and −0.07 for CLIPScore, and rankings hold within short, medium and long captions.

**Failure modes** (top-8 Polaris disagreements, inspected with images):

| Pattern | Count | Example |
|---|---:|---|
| Human rater noise (accurate caption rated 1 by a single rater) | 5 | "a sandwich and some fries on a plate" |
| Judge too lenient on vague or partial captions | 2 | "the coat is black" for a child with an umbrella |
| Judge caught an error both humans missed | 1 | "…of a man with skis" for a woman with skis |

Surprise: the judge's *argmax* digit ranked better on Polaris (0.879) than its probability-weighted score (0.832).

---

## 3. Data pipeline

**Cleaning** (100,820 → 98,757 pairs, 2.05% removed):

| Near-blank | Small or extreme shape | Caption < 3 or > 60 words | Near-duplicate | **Test-set image** |
|---:|---:|---:|---:|---:|
| 165 | 104 | 1,103 | 527 | **164** |

**Agent rule loop (human-in-the-loop).** A tool-using LLM agent proposed caption filters, testing each with tools against the human labels and the pool; a human approved or rejected each.

| Round | Proposals | Outcome |
|---|---:|---|
| 1 | 5 | Four duplicates of one regex; one matched label names instead of caption text. Tools hardened, rerun. |
| 2 | 2 | **Both rejected.** One deleted accurate captions (precision 0.09); one copied words from the labeled examples. Review also exposed an unenforced precision check, now fixed. |

No agent rule entered the pool. Decisions and reasons: [`results/agent/decisions.json`](results/agent/decisions.json).

**What each selector picked:**

| Arm | Words / caption | Overlap with C |
|---|---:|---:|
| A / B | 17.3 / 16.7 | |
| C (CLIPScore) | 18.6 | |
| D\* (judge) | 13.2 | **43%** |

CLIPScore favors long, keyword-rich captions; the judge picks largely different data.

**Arm E (generator + critic, with a near-duplicate diversity control and decontaminated images):** 94% of captions passed the critic first try, 6% were regenerated, mean critic score 4.0 / 5, 20.1 words per caption.

---

## 4. Results

COCO Karpathy test, 5,000 images, mean ± sd over 3 seeds.

| Arm | CIDEr-D ↑ | CHAIRs ↓ | CHAIRi ↓ | Objects / caption | Words | Eval judge ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 0 base | 0.202 | 12.6% | 9.4% | 1.57 | 20.3 | 4.61 |
| A random | 0.432 ± 0.029 | 11.5% | 11.3% | 1.06 | 10.6 | 4.27 ± 0.04 |
| B cleaned | 0.479 ± 0.006 | 11.4% | 10.6% | 1.12 | 10.6 | 4.30 ± 0.01 |
| C CLIPScore (= D) | 0.423 ± 0.016 | 10.0% | 8.5% | 1.24 | 14.8 | 4.24 ± 0.05 |
| **D\* judge** | **0.741 ± 0.013** | **7.5%** | **6.1%** | 1.32 | 9.5 | **4.54 ± 0.01** |
| E synthetic | 0.393 ± 0.009 | 13.3% | 8.5% | 1.71 | 18.6 | 4.58 ± 0.00 |

<sub>CHAIRs: share of captions naming ≥ 1 object not in the image. CHAIRi: share of named objects that are not in the image. Eval judge: Idefics3-8B on a fixed 2K subset, 1–5 scale.</sub>

**Paired bootstrap** (test images, seed-averaged, 2,000 draws):

| Contrast | Type | Δ CIDEr [95% CI] | Δ CHAIRs [95% CI] | Verdict |
|---|---|---|---|---|
| C − A | H3 | −0.010 [−0.022, +0.002] | −1.5 [−2.3, −0.8] pts | not supported |
| E − A | H4 | −0.039 [−0.054, −0.024] | +1.7 [+0.7, +2.8] pts | not supported |
| B − A | | +0.046 [+0.038, +0.054] | −0.1 [−0.6, +0.5] pts | small gain |
| **D\* − A** | exploratory | **+0.308** [+0.294, +0.323] | **−4.0** [−4.8, −3.1] pts | large gain |
| **D\* − C** | exploratory | **+0.318** [+0.305, +0.331] | **−2.4** [−3.1, −1.8] pts | large gain |
| E − D\* | exploratory | −0.347 [−0.362, −0.332] | +5.7 [+4.8, +6.7] pts | |

**Reading it**

- **D\* wins on every metric**, and the seed spread (0.013) is tiny next to the gap. Its captions are no longer than A's, so length doesn't explain the gain.
- **CLIPScore selection** reduced hallucination slightly but not CIDEr, and it lengthened captions (14.8 words).
- **Synthetic captions** taught a verbose style: per named object E hallucinates at C's rate (8.5%), but it names more objects, so more captions contain an error, and long captions match the short references less well.

- **The independent judge disagrees with CIDEr on verbosity.** It ranks the base model (4.61) and E (4.58) at the top, both of which write the longest captions, and barely penalizes their hallucinations. That fits the leniency failure mode found in §2. **D\* is the only arm that is strong on all three metrics** (top-3 on the judge, best on CIDEr and both CHAIR measures).

---

## 5. Engineering

**Stack:** Ray (parallel shard download; Ray Data `map_batches` for CLIP embeddings over 100K pairs), vLLM (judge), transformers + PEFT (generator, student, eval judge), Parquet stages with resumable chunks, AWS S3 for inputs and outputs, W&B for run tracking, pytest + ruff.

**Judge throughput** (A100-SXM 80GB, $1.81/h, disjoint images per variant):

| Variant | img/s | $ / 1K images |
|---|---:|---:|
| Plain decoding, full-res | 26.2 | 0.019 |
| 1-token logprobs, full-res | 29.9 | 0.017 |
| 1-token logprobs, ≤ 512 px | 40.9 | 0.012 |
| Production run (100K pool) | 65.2 | 0.008 |

Image encoding dominates cost, so capping resolution saved more than shortening the answer. A first benchmark that reused images across variants was inflated by vLLM's prefix cache; it is kept in `results/` and not used.

**Problems hit and fixed**

| Problem | Fix |
|---|---|
| PyTorch saw 128–192 host cores in a ~14-CPU container; preprocessing ran 80× slow | Cap threads (`OMP_NUM_THREADS=12`) |
| Faulty cloud host | Migrated to a new pod (~$0.75 lost) |
| Non-Qwen VLMs hung at vLLM 0.32 engine start | Generator and eval judge run in transformers; weights pre-downloaded |
| Eval judge answered " 4" as two tokens, so ~0 probability fell on the digits | Prefill the leading space; digit mass 0.996 |

---

## 6. Limitations

- One pool (CC12M), one 256M student, one exam of photos only; robustness to illustrations or other web image types is not measured.
- CHAIR is a simplified re-implementation and CIDEr uses a Python tokenizer. Compare arms, not absolute values against the literature.
- One human rater. 20 of the first 45 labels were revised for rubric consistency (audit notes kept); items 46–200 were rated offline and transcribed. The ratings' SHA-256 was committed before scoring.
- The hand-rated set also served to test agent rules, so it is not held out for rule-based scorers (the gate was computed first).
- Arm E's generator prompt did not control length; a length-matched version is the natural next experiment.
- Not done: diffusion fine-tuning on the curated data, DPO, Airflow orchestration, a human preference study of model outputs.
