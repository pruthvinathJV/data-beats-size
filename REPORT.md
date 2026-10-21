# Which 25K? Technical report

> **In plain words.** Most web captions are not descriptions. We compared five ways of picking 25,000 training examples
> for a small captioning model and gave each resulting model the same exam. Grading captions with a larger AI model,
> after checking its grades against human ones, beat every other method by a wide margin. A cheap similarity score
> did not help, and AI-rewritten captions made the model wordier rather than better.

This is a controlled ablation of data curation, down-selection and synthetic re-captioning: the model, training recipe and data budget are fixed, and only the data changes. The plan was written down before any results existed ([PREREGISTRATION.md](PREREGISTRATION.md)); every number below is in [`results/`](results/).

**Contents:** [TL;DR](#tldr) · [1 Setup](#1-setup) · [2 Can the judge be trusted?](#2-can-the-judge-be-trusted) · [3 Building the training sets](#3-building-the-training-sets) · [4 Results](#4-results) · [5 Engineering](#5-engineering) · [6 Limitations](#6-limitations)

---

## TL;DR

| | |
|---|---|
| **Best method** | Keep the 25% of web pairs a human-validated VLM judge rates highest: **CIDEr 0.432 → 0.741** (+71%) and **hallucinating captions 11.5% → 7.5%**, vs random data |
| **Why the judge, not CLIPScore** | On 200 hand-rated web captions the judge's ranking matches the human's twice as well (0.52 vs 0.25). On a clean public benchmark they look almost equal (0.83 vs 0.82) |
| **Did not help** | CLIPScore selection (CIDEr unchanged). AI-rewritten captions: an independent AI grader likes them (4.58 / 5), but CIDEr drops and hallucinated objects rise |
| **Robust winner** | Only the judge-selected arm is strong on all three measures |
| **Hidden leak** | 164 test images were in the training pool; removed before training |
| **Caveat** | The judge-selected arm was added after a planned checkpoint failed, so it is labeled exploratory (D\*) |
| **Cost** | $10.01: 5.5 hours on one A100 GPU at $1.81/h, including $0.75 lost on a faulty machine |

---

## 1. Setup

**The arms.** Each arm is a training set of 25K image–caption pairs. The same model is fine-tuned on each, three times with different random seeds.

| Arm | Training data | In the original plan? |
|---|---|---|
| 0 | none: the base model, as a reference | yes |
| A | random pairs from the raw pool | yes |
| B | random pairs after cleaning (filters, duplicate removal, test-image removal) | yes |
| C | cleaned pool, top 25K by CLIPScore | yes |
| D | cleaned pool, top 25K by the judge, *if* the judge passed a checkpoint (§2). It did not, so the plan switched D to CLIPScore, making it identical to C | yes |
| **D\*** | cleaned pool, top 25K by the judge, run anyway and labeled **exploratory** | added (amendment A2) |
| E | B's images with new captions from a generator model, checked by the judge as critic | added before any results (A1) |

**Data and models**

| Role | Choice |
|---|---|
| Pool | CC12M (web image–caption pairs), 100,820 pairs |
| Student (the model trained) | SmolVLM-256M-Instruct; LoRA r=16 on the language model (4.9M trainable weights, 2%); 1 epoch, lr 2e-4, batch 32 |
| Judge and critic | Qwen3-VL-8B-Instruct reads a written 1–5 rubric; its score is the average grade weighted by the probability it gives each digit |
| Caption generator (E) | InternVL3-2B |
| Independent grader | Idefics3-8B-Llama3, a model family not used to select or write any arm |
| Human labels | 200 CC12M pairs rated blind by the author with the same rubric; Polaris, 131K public human ratings |
| Exam | COCO Karpathy test split: 5,000 images, 5 human captions each |

**Measures**

| Measure | What it tells you |
|---|---|
| CIDEr-D | how closely the model's captions match the human captions (higher is better) |
| CHAIRs / CHAIRi | share of captions that name an object not in the image / share of named objects that are not in the image (lower is better) |
| Independent grader | Idefics3's 1–5 grade of each caption, on 2,000 test images |
| Spearman (rank agreement) | how well a scorer orders captions the way humans do: 1 = identical order, 0 = unrelated |

---

## 2. Can the judge be trusted?

Grading 100K pairs with the judge is the expensive step, so the plan required the judge to first beat CLIPScore against human ratings (the **checkpoint**): by at least 0.05 in rank agreement on Polaris, and with no loss on the hand-rated web set.

| Human ratings | Judge | CLIPScore | Difference [95% CI] | Checkpoint |
|---|---:|---:|---|---|
| Polaris test (8,726 clean captions) | 0.832 | 0.819 | +0.013 [0.010, 0.015] | ✗ below 0.05 |
| Hand-rated web captions (200) | **0.518** | 0.246 | **+0.272** [0.135, 0.414] | ✓ |

→ **Failed as planned (NO-GO)**, so arm D fell back to CLIPScore. Yet the judge's advantage is 20× larger on real web captions than on the clean benchmark, which is why D\* was run as an exploratory arm.

**Calibration.** The judge's raw grades run off the human scale; a monotone mapping (isotonic regression, fit on Polaris train) fixes the scale without changing the ranking. MAE = average error in grade points; ECE = average gap between predicted and actual human grade across score bins.

| | MAE | ECE |
|---|---:|---:|
| Judge, raw | 0.468 | 0.394 |
| Judge, calibrated | **0.210** | **0.047** |
| CLIPScore, calibrated | 0.271 | 0.077 |

**How good can agreement get?** Two humans rating the same caption agree at only **0.255**. The judge agrees with a single human at **0.328**: it behaves like the *average* rater.

**Length bias:** none found. Correlation with caption length is −0.03 for the judge and −0.07 for CLIPScore, and rankings hold within short, medium and long captions.

**Where judge and humans disagree most** (top 8 on Polaris, checked against the images):

| Pattern | Count | Example |
|---|---:|---|
| The human was wrong: an accurate caption rated 1 by a single rater | 5 | "a sandwich and some fries on a plate" |
| The judge was too lenient on a vague or partial caption | 2 | "the coat is black" for a child with an umbrella |
| The judge caught an error both humans missed | 1 | "…of a man with skis" for a woman with skis |

One surprise: the judge's single most likely grade ranked captions better on Polaris (0.879) than its probability-weighted grade (0.832). Argmax would have cleared the Polaris part of the gate (0.879 vs CLIPScore's 0.819); it was never scored on the hand-rated set. It was not adopted: the scoring rule was pre-registered, switching after seeing results is a forking path, and argmax's five tied grade buckets are poorly suited to top-k selection.

---

## 3. Building the training sets

**Cleaning:** 100,820 → 98,757 pairs (2.05% removed).

| Near-blank image | Too small or odd shape | Caption < 3 or > 60 words | Near-duplicate | **Test-set image** |
|---:|---:|---:|---:|---:|
| 165 | 104 | 1,103 | 527 | **164** |

**AI agent with a human checkpoint.** An LLM agent proposed caption filter rules. It could only act through tools that test a rule against the human ratings and show what it would remove from the pool. A human approved or rejected each proposal.

| Round | Proposals | Outcome |
|---|---:|---|
| 1 | 5 | Four were the same rule under different names; one searched for the human's label names instead of caption text. The tools were tightened and the agent rerun. |
| 2 | 2 | **Both rejected.** One would delete mostly accurate captions (only 9% of its hits were rated bad); the other copied words from the rated examples. The review also exposed a missing precision check in the tools, now fixed. |

No agent rule was used. Decisions and reasons: [`results/agent/decisions.json`](results/agent/decisions.json).

**What each method picked:**

| Arm | Words per caption | Overlap with C |
|---|---:|---:|
| A / B | 17.3 / 16.7 | |
| C (CLIPScore) | 18.6 | |
| D\* (judge) | 13.2 | **43%** |

CLIPScore prefers long, keyword-heavy captions; the judge picks mostly different data.

**Arm E (generator + critic).** InternVL3-2B wrote one sentence per image; the judge graded each; captions below 3.5 / 5, or near-copies of an earlier caption (a diversity control), were rewritten once. 94% passed first time, 6% were rewritten; the average grade was 4.0 / 5 and the average length 20.1 words.

---

## 4. Results

5,000 COCO test images; mean ± standard deviation over 3 training runs.

| Arm | CIDEr-D ↑ | CHAIRs ↓ | CHAIRi ↓ | Objects / caption | Words | Independent grader ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 0 base | 0.202 | 12.6% | 9.4% | 1.57 | 20.3 | 4.61 |
| A random | 0.432 ± 0.029 | 11.5% | 11.3% | 1.06 | 10.6 | 4.27 ± 0.04 |
| B cleaned | 0.479 ± 0.006 | 11.4% | 10.6% | 1.12 | 10.6 | 4.30 ± 0.01 |
| C CLIPScore (= D) | 0.423 ± 0.016 | 10.0% | 8.5% | 1.24 | 14.8 | 4.24 ± 0.05 |
| **D\* judge** | **0.741 ± 0.013** | **7.5%** | **6.1%** | 1.32 | 9.5 | **4.54 ± 0.01** |
| E synthetic | 0.393 ± 0.009 | 13.3% | 8.5% | 1.71 | 18.6 | 4.58 ± 0.00 |

**Head-to-head comparisons** (paired bootstrap over test images, 2,000 resamples; the planned hypotheses are H3 "CLIPScore beats random" and H4 "synthetic beats random"):

| Comparison | Type | Δ CIDEr [95% CI] | Δ CHAIRs [95% CI] | Verdict |
|---|---|---|---|---|
| C − A | planned (H3) | −0.010 [−0.022, +0.002] | −1.5 [−2.3, −0.8] pts | no CIDEr gain |
| E − A | planned (H4) | −0.039 [−0.054, −0.024] | +1.7 [+0.7, +2.8] pts | worse |
| B − A | planned | +0.046 [+0.038, +0.054] | −0.1 [−0.6, +0.5] pts | small gain |
| **D\* − A** | exploratory | **+0.308** [+0.294, +0.323] | **−4.0** [−4.8, −3.1] pts | large gain |
| **D\* − C** | exploratory | **+0.318** [+0.305, +0.331] | **−2.4** [−3.1, −1.8] pts | large gain |
| E − D\* | exploratory | −0.347 [−0.362, −0.332] | +5.7 [+4.8, +6.7] pts | D\* better |

**Reading it**

- **D\* wins clearly.** Best CIDEr and lowest hallucination, and the spread across its 3 runs (0.013) is tiny next to the gap. Its captions are no longer than A's, so the gain is not about length.
- **What the CIs cover.** The paired bootstrap resamples test images, so it captures exam noise, not training noise. Training noise is the ± over 3 seeds: the D\*−A gap (+0.308) is about 10× the largest seed SD (0.029).
- **CLIPScore selection** reduced hallucination a little but did not raise CIDEr, and made captions longer (14.8 words).
- **AI-rewritten captions (E)** taught the model a wordy style. Per named object it hallucinates no more than C (8.5%), but it names more objects, so more captions contain a mistake, and long captions match the short human references less well.
- **The independent grader favors long captions.** It ranks the base model (4.61) and E (4.58) highest, the two wordiest, and barely penalizes their hallucinations, matching the leniency seen in §2. **D\* is the only arm strong on all three measures.**

---

## 5. Engineering

**Stack:** Ray (parallel downloads; Ray Data `map_batches` for CLIP embeddings over 100K pairs), vLLM (judge), transformers + PEFT (generator, student, independent grader), Parquet stages that resume after interruption, AWS S3 for storage, W&B for run tracking, pytest + ruff.

**Making the judge cheaper** (A100 80GB at $1.81/h; each variant measured on different images):

| Variant | Images / s | $ per 1K images |
|---|---:|---:|
| Write the answer normally, full-size image | 26.2 | 0.019 |
| Read one token's probabilities, full-size image | 29.9 | 0.017 |
| Read one token's probabilities, image capped at 512 px | 40.9 | 0.012 |
| Production run (100K pool) | 65.2 | 0.008 |

Encoding the image is the main cost, so shrinking images saved more than shortening the answer. An earlier benchmark reused the same images across variants and was flattered by vLLM's cache; it is kept in `results/` for transparency and not used.

**Problems hit and fixed**

| Problem | Fix |
|---|---|
| PyTorch saw 128–192 CPU cores in a container limited to ~14; image preprocessing ran 80× slower | cap threads (`OMP_NUM_THREADS=12`) |
| Faulty cloud machine | moved to a new one (~$0.75 lost) |
| Non-Qwen models froze at start-up in vLLM 0.32 | run the generator and independent grader in transformers; download weights beforehand |
| The independent grader answers " 4" as two tokens, so almost no probability landed on the digits | add the leading space to the prompt; 99.6% of probability now on the digits |

---

## 6. Limitations

- One data source (CC12M), one small model, one photo-only exam (COCO). Judge-selected captions are shorter and plainer than CLIPScore's (13.2 vs 18.6 words), closer to COCO's style, so part of D\*'s gain may be style match; an out-of-domain exam (Flickr30k or nocaps) is the next check. No larger model was compared, so the study says nothing about curation vs. model size. How filtering affects illustrations or other image types is not measured.
- CHAIR is a simplified re-implementation and CIDEr uses a Python tokenizer: compare arms with each other, not with published numbers.
- One human rater. 20 of the first 45 ratings were revised for consistency (audit notes kept); ratings 46–200 were made offline and transcribed. A fingerprint (SHA-256) of the ratings was committed before any scoring.
- The hand-rated set was also used to test the agent's rules, so it is not untouched data for rule-based scorers (the checkpoint was computed first).
- Arm E's generator was not asked for short captions; a length-controlled version is the natural next experiment.
- Not done: training an image generator (diffusion) on the curated data, preference tuning (DPO), workflow orchestration (Airflow), a human comparison of model outputs.
