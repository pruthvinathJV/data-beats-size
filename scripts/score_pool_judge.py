"""Stage 4a: selection judge over the whole raw pool.

Scoring every raw pair (not just the filtered B pool) lets this run while the
filter rules are still under review; arms only use scores of B-pool rows.
Resumable: one parquet per chunk under data/pool/judge_chunks/.

    python scripts/score_pool_judge.py
"""

import io
import json
import time

import pandas as pd
from PIL import Image

from datacurate.io import DATA, RESULTS, load_config
from datacurate.judge import VLMJudge
from datacurate.scoring import throughput

CHUNK = 4096


def main():
    cfg = load_config("configs/default.yaml")
    jc = cfg["judge"]
    b = pd.read_parquet(DATA / "pool" / "raw", columns=["key", "caption"]).reset_index(drop=True)
    raw = pd.read_parquet(DATA / "pool" / "raw", columns=["key", "img"]).set_index("key").img
    judge = VLMJudge(jc["model"], jc["rubric"], max_image_side=jc["max_image_side"],
                     max_model_len=jc["max_model_len"], gpu_memory_utilization=jc["gpu_memory_utilization"],
                     top_logprobs=jc["top_logprobs"], seed=cfg["seed"])
    out = DATA / "pool" / "judge_chunks"
    out.mkdir(parents=True, exist_ok=True)
    n_new, secs = 0, 0.0
    for s in range(0, len(b), CHUNK):
        path = out / f"chunk_{s // CHUNK:04d}.parquet"
        if path.exists():
            continue
        rows = b.iloc[s : s + CHUNK]
        imgs = [Image.open(io.BytesIO(raw[k])) for k in rows.key]
        t0 = time.perf_counter()
        sc = judge.score(imgs, rows.caption.tolist())
        secs += time.perf_counter() - t0
        n_new += len(rows)
        pd.DataFrame({"key": rows.key.to_numpy(), "judge": [x.expected for x in sc],
                      "digit_mass": [x.digit_mass for x in sc]}).to_parquet(path)
        print(f"{s + len(rows):,}/{len(b):,}", flush=True)
    scores = pd.read_parquet(out)
    scores.to_parquet(DATA / "pool" / "judge_scores.parquet")
    tp = throughput(n_new, secs, cfg["cost"]["usd_per_hour"]) if n_new else {}
    (RESULTS / "pool_judge.json").write_text(json.dumps({
        "scored": len(scores), "parse_failure_rate": round(float((scores.digit_mass < 0.5).mean()), 4),
        "rubric_sha256": judge.rubric_sha, **tp}, indent=2) + "\n")
    print(f"{len(scores):,} judge scores; {tp}")


if __name__ == "__main__":
    main()
