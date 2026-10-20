"""10-minute go/no-go for the selection judge, plus the throughput table.

Checks: digits "1".."5" are single tokens, parse-failure rate on N Polaris
test rows, latency. With --benchmark, also measures images/s and $ per 1K
images for plain decoding vs the optimized 1-token logprob path.

    python scripts/smoke_test_judge.py --benchmark
    python scripts/smoke_test_judge.py --model Qwen/Qwen2.5-VL-7B-Instruct

Exit code 1 means: set judge.model to judge.fallback_model and rerun.
"""

import argparse
import io
import json
import re
import time

import numpy as np
from PIL import Image

from datacurate.io import RESULTS, load_config  # loads .env first
from datacurate.judge import VLMJudge
from datacurate.polaris import collapse_ratings, load_polaris
from datacurate.scoring import throughput
from datacurate.stats import corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/polaris_calibration.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    jc, sc = cfg["judge"], cfg["smoke_test"]
    model = args.model or jc["model"]
    judge = VLMJudge(
        model,
        jc["rubric"],
        max_image_side=jc["max_image_side"],
        max_model_len=jc["max_model_len"],
        gpu_memory_utilization=jc["gpu_memory_utilization"],
        top_logprobs=jc["top_logprobs"],
        seed=cfg["seed"],
    )

    token_ids = judge.digit_token_ids()
    single_tokens = all(len(ids) == 1 for ids in token_ids.values())

    n = sc["n_rows"] + (3 * sc["bench_rows"] if args.benchmark else 0)
    pairs = collapse_ratings(load_polaris("test", n_shards=1)).sample(n=n, random_state=cfg["seed"])
    images = [Image.open(io.BytesIO(b)) for b in pairs.img_bytes]
    caps = pairs.cand.tolist()

    k = sc["n_rows"]
    t0 = time.perf_counter()
    scores = judge.score(images[:k], caps[:k])
    latency = time.perf_counter() - t0
    fail_rate = float(np.mean([not s.parse_ok for s in scores]))
    expected = np.array([s.expected for s in scores])

    report = {
        "model": model,
        "rubric_sha": judge.rubric_sha,
        "digit_token_ids": token_ids,
        "single_token_digits": single_tokens,
        "n_rows": k,
        "parse_failure_rate": fail_rate,
        "mean_digit_mass": float(np.mean([s.digit_mass for s in scores])),
        "argmax_histogram": np.bincount([s.argmax for s in scores], minlength=6)[1:].tolist(),
        "spearman_vs_human_sanity": corr(expected, pairs.human.to_numpy()[:k]),  # n=50: sign check only
        "batch_latency_s": latency,
    }

    if args.benchmark:
        report["throughput"] = benchmark(judge, images[k:], caps[k:], cfg["cost"]["usd_per_hour"])

    report["passed"] = single_tokens and fail_rate <= sc["max_parse_failure_rate"]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"smoke_{model.split('/')[-1]}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print(f"\nFAILED. Set judge.model to {jc['fallback_model']} and rerun.")
    raise SystemExit(0 if report["passed"] else 1)


def benchmark(judge: VLMJudge, images, caps, usd_per_hour) -> list[dict]:
    """Each variant gets its own disjoint rows, so vLLM's prefix cache cannot carry
    encoded images from one variant into the next and flatter it."""
    third = len(images) // 3
    parts = [(images[i * third : (i + 1) * third], caps[i * third : (i + 1) * third]) for i in range(3)]

    def plain(im, cp):
        return [bool(re.search(r"[1-5]", o)) for o in judge.generate_text(im, cp, max_side=None)]

    variants = [
        ("plain decode, full-res", plain),
        ("1-token logprobs, full-res", lambda im, cp: [s.parse_ok for s in judge.score(im, cp, max_side=None)]),
        (f"1-token logprobs, {judge.max_image_side}px", lambda im, cp: [s.parse_ok for s in judge.score(im, cp)]),
    ]
    rows = []
    for (name, fn), (im, cp) in zip(variants, parts):
        t0 = time.perf_counter()
        ok = fn(im, cp)
        dt = time.perf_counter() - t0
        rows.append({"variant": name, "n": len(im), "seconds": dt, "parse_ok": float(np.mean(ok)),
                     **throughput(len(im), dt, usd_per_hour)})
    return rows


if __name__ == "__main__":
    main()
