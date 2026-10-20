# Data beats size

Curate a noisy web image-caption pool (CC12M) four ways, LoRA fine-tune the same
small VLM (SmolVLM-256M) on each 50K-example arm, and measure which curation
method improves captioning on COCO. The judge used for curation is checked
against human labels (Polaris) before it is trusted.

Hypotheses, metrics and the decision rule for the expensive judge run are fixed
in [PREREGISTRATION.md](PREREGISTRATION.md).

## Status

| Step | State |
|---|---|
| 0. Blind hand-rating set (200 CC12M pairs) | done (fingerprint in results/handrate_manifest.json) |
| 1. Judge smoke test + throughput table | ready to run |
| 2. Polaris judge calibration | ready to run |
| 3. Pool: filter, embed, dedup, decontaminate | not started |
| 4. Arms A-D | not started |
| 5. LoRA training (arm 0 + 4 arms x 3 seeds) | not started |
| 6. Karpathy-test eval | not started |

## Layout

```
configs/      run configs (one YAML per stage)
prompts/      versioned judge rubric; its hash is logged with every score
src/datacurate/
  judge.py        vLLM judge, probability-weighted 1-5 score from 1-token logprobs
  clipscore.py    CLIP embeddings + CLIPScore
  polaris.py      Polaris loading, rating collapse, rater pairs for the human ceiling
  stats.py        rank correlations, image-clustered bootstrap, pre-registered gate
  calibration.py  isotonic fit, ECE, MAE, reliability table
  scoring.py      chunked, resumable scoring loop; throughput and $ per 1K images
scripts/      entry points (see runbook)
notebooks/    analysis; reads scored parquet, no GPU needed
tests/        unit tests for parsing, stats, calibration
```

## Runbook

**Local (laptop), before touching a GPU:**

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,notebook]"
cp .env.example .env            # fill in keys
python scripts/make_handrate_set.py
python scripts/rate_server.py             # keep running; open http://localhost:8765/pages/batch_01.html
                                          # 14 pages x 15 items; every change saves to data/handrate/ratings/
python scripts/merge_handrate.py          # -> data/handrate/ratings.csv, prints progress
git add PREREGISTRATION.md && git commit -m "Pre-register judge gate and hypotheses"
```

On macOS with an iCloud-synced Desktop, sync keeps marking the venv's `.pth` files hidden and Python
skips them (`import datacurate` fails). Keep the venv outside the synced folder and symlink it:
`uv venv --python 3.12 ~/.venvs/data-beats-size && ln -s ~/.venvs/data-beats-size .venv`.

Copy `data/handrate/` to the pod (it is gitignored), e.g. `aws s3 sync data/handrate s3://$S3_BUCKET/handrate`.

**RunPod (Secure Cloud, 1x A100 80GB; L40S 48GB fallback):**

```bash
cd /workspace && git clone <this repo> && cd <repo>
cp .env.example .env            # fill in keys; uncomment HF_HOME=/workspace/hf_cache
bash scripts/pod_setup.sh && source .venv/bin/activate
python scripts/smoke_test_judge.py --benchmark   # exit 1 => switch to fallback model in config
python scripts/score_polaris.py                  # resumable; rerun after a restart
python scripts/score_handrate.py                 # refuses to run without your ratings
```

Then pull `data/scores/` back (S3) and run `notebooks/01_polaris_calibration.ipynb` locally.
It writes `results/polaris_gate.json` with the GO / NO-GO decision.

Set `cost.usd_per_hour` in the config from the RunPod console so throughput is reported in $ per 1K images.
Stop the pod when a step finishes.
