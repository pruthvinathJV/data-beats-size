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

(none)
