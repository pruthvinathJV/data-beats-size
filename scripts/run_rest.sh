#!/usr/bin/env bash
# Stages 2c-7 after the gate and the author's rule approvals, in order.
# Stops at the first failure; every stage is resumable, so rerunning skips finished work.
#   bash scripts/run_rest.sh > /workspace/logs/run_rest.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
# Containers report the host's core count; cap threads to the container's CPU quota.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12} TOKENIZERS_PARALLELISM=false
stage() { echo "=== $(date -u +%T) $*"; }

stage filter;        python scripts/filter_pool.py
# Judge scores are needed for D (if GO) or the exploratory D* (A2), and for nothing else.
stage judge-pool;    python scripts/score_pool_judge.py
stage arms;          python scripts/make_arms.py
for phase in generate critique retry critique2 finalize; do
  if [ "$phase" = generate ] && [ -f data/arms/e_work/gen1.parquet ]; then continue; fi
  stage "arm-E $phase"; python scripts/recaption.py --phase "$phase"
done

stage train
# Three runs share the GPU at a time; SmolVLM-256M with LoRA needs only a few GB each.
jobs=()
for seed in 0 1 2; do
  # D is identical to C after the NO-GO gate (A2), so C stands for it; Dx is exploratory D*.
  for arm in A B C Dx E; do jobs+=("$arm $seed"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P 3 -L 1 bash -c 'python scripts/train_lora.py --arm "$0" --seed "$1" > "/workspace/logs/train_$0_s$1.log" 2>&1 || { echo "train $0 s$1 failed"; exit 255; }'

stage eval-generate; python scripts/eval_generate.py
stage eval-metrics;  python scripts/eval_metrics.py
stage eval-judge;    python scripts/eval_judge.py
stage upload;        python -c "from datacurate.io import RESULTS, DATA, upload_to_s3; print(upload_to_s3(RESULTS, 'outputs'), upload_to_s3(DATA / 'eval', 'outputs'))"
stage DONE
