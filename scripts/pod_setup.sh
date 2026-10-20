#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (PyTorch template, CUDA 12.x).
# Clone the repo under /workspace (the persistent volume), then: bash scripts/pod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

test -f .env || { echo "Create .env from .env.example first (uncomment HF_HOME=/workspace/hf_cache)."; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12
uv pip install -e ".[gpu,dev]"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Ready. Next: source .venv/bin/activate && python scripts/smoke_test_judge.py --benchmark"
