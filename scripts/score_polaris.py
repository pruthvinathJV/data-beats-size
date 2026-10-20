"""Score Polaris (train subset + full test) with the judge and CLIPScore.

Writes data/scores/polaris_{split}.parquet (no image bytes) and, for test,
data/scores/polaris_test_thumbs.parquet (256px thumbnails for the
disagreement review). Resumable: rerun the same command after a restart.

    python scripts/score_polaris.py
"""

import argparse
import io
import json

import pandas as pd
from PIL import Image

from datacurate.clipscore import CLIPScorer
from datacurate.io import DATA, RESULTS, init_wandb, load_config, upload_to_s3  # loads .env first
from datacurate.judge import VLMJudge, resize_max_side
from datacurate.polaris import collapse_ratings, load_polaris
from datacurate.scoring import score_rows, throughput


def thumbnail_bytes(b: bytes, side: int = 256) -> bytes:
    buf = io.BytesIO()
    resize_max_side(Image.open(io.BytesIO(b)), side).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/polaris_calibration.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    jc = cfg["judge"]
    judge = VLMJudge(
        jc["model"],
        jc["rubric"],
        max_image_side=jc["max_image_side"],
        max_model_len=jc["max_model_len"],
        gpu_memory_utilization=jc["gpu_memory_utilization"],
        top_logprobs=jc["top_logprobs"],
        seed=cfg["seed"],
    )
    clip = CLIPScorer(cfg["clip"]["model"], batch_size=cfg["clip"]["batch_size"])
    run = init_wandb(cfg, "score-polaris", rubric_sha=judge.rubric_sha)

    out = DATA / "scores"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"judge_model": jc["model"], "rubric_sha": judge.rubric_sha, "gpu": cfg["cost"]["gpu"]}
    for split in args.splits:
        n_shards = cfg["polaris"]["train_shards"] if split == "train" else None
        pairs = collapse_ratings(load_polaris(split, n_shards))

        if split == "test":
            imgs = pairs.drop_duplicates("img_hash")
            pd.DataFrame(
                {"img_hash": imgs.img_hash, "thumb": imgs.img_bytes.map(thumbnail_bytes)}
            ).to_parquet(out / "polaris_test_thumbs.parquet")

        scores, timing = score_rows(
            pairs,
            lambda r: Image.open(io.BytesIO(r.img_bytes)),
            judge,
            clip,
            out / f"polaris_{split}_chunks",
            jc["chunk_size"],
        )
        scored = pairs.drop(columns="img_bytes").join(scores)
        scored.to_parquet(out / f"polaris_{split}.parquet")

        stats = {
            "rows": len(scored),
            "unique_images": scored.img_hash.nunique(),
            "parse_failure_rate": float((scored.digit_mass < 0.5).mean()),
            **timing,
            "judge": throughput(timing["n"], timing["judge_s"], cfg["cost"]["usd_per_hour"]),
            "clip": throughput(timing["n"], timing["clip_s"], cfg["cost"]["usd_per_hour"]),
        }
        summary[split] = stats
        run.log({f"{split}/{k}": v for k, v in stats.items() if isinstance(v, int | float)})
        print(split, json.dumps(stats, indent=2))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "polaris_scoring.json").write_text(json.dumps(summary, indent=2))
    print("uploaded:", upload_to_s3(out, "polaris"))
    run.finish()


if __name__ == "__main__":
    main()
