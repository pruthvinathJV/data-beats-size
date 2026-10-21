"""Score the hand-rated CC12M set with the judge and CLIPScore.

Guardrail for blind labeling: refuses to run until data/handrate/ratings.csv
exists with at least 150 ratings, so model scores can never be seen first.

    python scripts/score_handrate.py
"""

import argparse
import json

import pandas as pd
from PIL import Image

from datacurate.clipscore import CLIPScorer
from datacurate.io import DATA, RESULTS, load_config, upload_to_s3  # loads .env first
from datacurate.judge import VLMJudge
from datacurate.scoring import score_rows

MIN_RATED = 150  # below this the paired CI is too wide for the gate to mean much


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    d = DATA / "handrate"
    ratings_path = d / "ratings.csv"
    if not ratings_path.exists():
        raise SystemExit(f"{ratings_path} not found. Rate data/handrate/pages/ and run merge_handrate.py first.")
    items = pd.read_parquet(d / "items.parquet")
    ratings = pd.read_csv(ratings_path, dtype={"key": str}).dropna(subset=["rating"])
    if len(ratings) < MIN_RATED:
        raise SystemExit(f"Only {len(ratings)} items rated; need {MIN_RATED} before scoring.")
    df = items.merge(ratings[["key", "rating"]], on="key")

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
    scores, _ = score_rows(
        df,
        lambda r: Image.open(d / "images" / f"{r.key}.jpg"),
        judge,
        clip,
        DATA / "scores" / "handrate_chunks",
        jc["chunk_size"],
        caption_col="caption",
    )
    df.join(scores).assign(n_words=lambda x: x.caption.str.split().str.len()).to_parquet(
        DATA / "scores" / "handrate.parquet"
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "handrate_scoring.json").write_text(
        json.dumps({"n": len(df), "judge_model": jc["model"], "rubric_sha": judge.rubric_sha}, indent=2)
    )
    print("uploaded:", upload_to_s3(DATA / "scores" / "handrate.parquet", "handrate"))


if __name__ == "__main__":
    main()
