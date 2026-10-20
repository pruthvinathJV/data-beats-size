# Pre-registration

Written before any judge, CLIPScore, or training result exists. The git commit
timestamp of this file is the record. Changes after results exist go in the
Amendments section with a date and a reason; nothing above it is edited.

## Questions

1. **Judge quality.** Does a probability-weighted VLM judge rank image-caption quality closer to humans than CLIPScore?
2. **Curation.** At a fixed 50K-example budget, does curation (dedup, CLIPScore top-k, judge top-k) improve a small VLM's captioning over random selection?

## Setup (fixed)

- **Pool:** CC12M (`pixparse/cc12m-wds`), shards 0000-0039 (~200K pairs).
- **Hand-rated set:** 200 pairs sampled from shard 2175 (outside the pool), at least 150 rated blind (in presentation order) by the author against `prompts/judge_rubric_v1.md` before any model score for them exists. All 200 were rated. Items 1-45 were rated in the rating page; items 46-200 were rated offline and transcribed into the rating files, so their `rated_at` values are transcription times. 20 of items 1-45 were revised for consistency after a rubric review, each with an audit note in the ratings file. The SHA-256 of the final ratings file is in `results/handrate_manifest.json`, committed before any model scoring.
- **Selection judge:** Qwen3-VL-8B-Instruct; fallback Qwen2.5-VL-7B-Instruct if the smoke test fails (single-token digits, parse failures <= 2%). Score = expected value of the 1-5 digit distribution from 1-token logprobs.
- **CLIPScore:** `openai/clip-vit-large-patch14`, 2.5 * max(cos, 0).
- **Arms (50K each):** 0 = base model, no fine-tune; A = random from pool; B = random from heuristic-filtered, deduplicated, decontaminated pool; C = B pool, CLIPScore top-k; D = B pool, judge top-k.
- **Model:** SmolVLM-256M-Instruct, LoRA, identical hyperparameters and steps per arm, 3 seeds per arm. Fixed prompt: "Describe this image in one sentence." for training and eval.
- **Eval:** COCO Karpathy test (5K images).

## Gate for the full judge run (~150K pairs)

Run the judge on the full pool **only if both hold**:

1. On Polaris test, judge Spearman vs mean human score exceeds CLIPScore Spearman by **>= 0.05**, with **non-overlapping 95% CIs** (bootstrap resampling images, 2000 draws).
2. On the hand-rated CC12M set, the judge is **at least tied** with CLIPScore: the upper bound of the paired 95% CI on (judge - CLIPScore) Spearman is >= 0.

If the gate fails, arm D becomes "CLIPScore top-k with judge as tiebreaker" and the README reports why.

## Judge metrics (reported regardless of the gate)

- Spearman and Kendall vs human, with 95% CIs, for: judge expected score, judge argmax digit, CLIPScore.
- Human ceiling: Spearman between two different raters of the same pair (pairs with 2+ ratings), next to judge-vs-single-rater on the same pairs.
- Calibration: isotonic fit on Polaris train, applied to test. MAE and ECE (10 equal-mass bins) before and after. Rank metrics are not used for calibration, since isotonic maps are monotonic.
- Length bias: Spearman within caption-length tertiles, and corr(score, length) for judge, CLIPScore, and humans.
- 20 largest calibrated-judge vs human disagreements, labeled by failure mode.

## Curation hypotheses

- **H1:** D > A on CIDEr (Karpathy test).
- **H2:** D >= C on CIDEr. A tie is a valid result.
- **H3:** C > A on CIDEr.

**Primary metrics:** CIDEr; an uncalibrated judge from a non-Qwen family (InternVL or Gemma); CHAIR (object hallucination, lower is better).
**Secondary:** Polos. It is trained on Polaris labels, which the selection judge is calibrated on, so it is not independent.
**Statistics:** mean over 3 seeds; paired bootstrap over test images for D-A and D-C; report effect sizes with 95% CIs.
**Confounders reported:** mean caption length per arm (training data and generated outputs); decontamination removals vs Karpathy test.

## Amendments

### A1 (2026-10-20, before any model scoring): scope for a 1-2 day, ~$25-30 budget; synthetic and agentic arms

Reason: extend the study to two further data interventions, synthetic re-captioning and agent-proposed filter rules with human approval, while fitting a fixed time and compute budget. Made before any judge, CLIPScore, or training output exists. Where this section conflicts with the sections above, this section governs.

**Pool and arm size.** Pool = CC12M shards 0000-0019 (~100K pairs). Arms are **25K examples each** (was 50K), keeping a ~25-35% keep rate.

**Arms.**
- 0: base SmolVLM-256M-Instruct, no fine-tune.
- A: 25K random from the raw pool.
- B: 25K random from the B pool = raw pool after heuristic filters (incl. near-blank image filter), the agent-proposed rules the author approved (below), CLIP-embedding near-duplicate removal, and decontamination against Karpathy test images.
- C: B pool, CLIPScore top 25K.
- D: B pool, judge top 25K (if the gate fails: CLIPScore top 25K re-ranked by judge as tiebreaker, as specified above).
- **E (new): synthetic captions.** The 25K images of arm B, re-captioned by a generator (InternVL3-2B) with the fixed prompt "Describe this image in one sentence.". A critic (the selection judge) scores each caption; captions below the critic threshold (calibrated score < 3.5) are regenerated once at a higher temperature, then the better of the two is kept. Diversity control: a caption that is a near-duplicate (token Jaccard >= 0.9) of an already-accepted caption is regenerated once. Contamination control: arm B images are already decontaminated. Reported: acceptance rate, retry rate, caption length, distinct-n.

**Model families.** Selection judge and arm E critic: Qwen3-VL-8B-Instruct. Arm E generator: InternVL3-2B. Evaluation judge: Phi-3.5-vision-instruct. Three different families so no model scores text from its own family. (Gemma 3 was considered and dropped: gated with manual approval.)

**Agentic human-in-the-loop rule loop.** An LLM agent reads judge-vs-human disagreements and the author's failure-mode tags, proposes caption filter rules, and tests each with tools (agreement with the author's labels, share of the pool removed). The author approves or rejects each rule with a written reason; only approved rules enter the B pool. All proposals, test results and decisions are logged. Because the author's labels are used to test rules, the hand-rated set is no longer a clean held-out check for any rule-based scorer; the judge-vs-CLIPScore gate on it is computed before the agent sees it.

**Training.** Arms A-E x 3 seeds = 15 LoRA runs, identical hyperparameters and step counts.

**Evaluation.** Karpathy test, 5K images: CIDEr and CHAIR (CHAIRi, CHAIRs) on all images; evaluation judge on a fixed random 2K subset. Polos is dropped (time). Hypotheses H1-H3 unchanged, with 25K arms. New:
- **H4:** E > A on CIDEr.
- **H5 (exploratory, no direction):** E vs D on CIDEr and CHAIR.
Mean caption length per arm is reported for every arm, since synthetic captions may differ in length and style from web captions.

**Cut (named as limitations and future work):** diffusion fine-tuning, DPO, Airflow DAG, human pairwise preference study, LoRA vs QLoRA table, 2B confirmation run.

**Budget guard.** Long-running GPU jobs stop the pod on completion.

### A2 (2026-10-20, after the gate, before any arm is built or any model is trained)

**Gate outcome (confirmatory, unchanged).** On Polaris test, judge Spearman 0.832 vs CLIPScore 0.819: gain +0.013 (paired 95% CI 0.010-0.015), below the pre-registered 0.05. On the hand-rated CC12M set the judge was at least tied (judge - CLIPScore +0.272, CI 0.135-0.414). Decision: **NO-GO**. Per the fallback, arm D = CLIPScore top-k with the judge as tiebreaker, which is the same data as arm C up to ties; arm D is therefore not trained separately, and arm C's results stand for it.

**Exploratory arm D\* (new, labeled exploratory everywhere it is reported).** D\* = B pool, judge top 25K. Motivation: the in-domain hand-rated result (+0.27) suggests the judge's advantage over CLIPScore is much larger on noisy web captions than on Polaris; the confirmatory gate did not require testing that. D\* is trained and evaluated exactly like the other arms (3 seeds). Its contrasts (D\* - A, D\* - C, E - D\*) are reported as exploratory and never as the pre-registered H1/H2.

**Unchanged:** all thresholds, arms A, B, C, E, the arm E critic (the selection judge, as in A1), metrics and statistics.
