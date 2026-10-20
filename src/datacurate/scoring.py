"""Shared scoring loop: judge + CLIPScore over (image, caption) rows.

Work is chunked and each chunk is written to its own parquet file, so a pod
restart resumes where it stopped instead of rescoring.
"""

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def score_rows(
    df: pd.DataFrame,
    load_image: Callable[[pd.Series], Image.Image],
    judge,
    clip,
    out_dir: Path,
    chunk_size: int,
    caption_col: str = "cand",
) -> tuple[pd.DataFrame, dict]:
    """Returns per-row scores aligned to df.index, and timing for chunks scored in this call."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {"n": 0, "judge_s": 0.0, "clip_s": 0.0, "parse_failures": 0}
    chunks = []
    for start in tqdm(range(0, len(df), chunk_size), desc=out_dir.name):
        path = out_dir / f"chunk_{start // chunk_size:05d}.parquet"
        if path.exists():
            chunks.append(pd.read_parquet(path))
            continue
        rows = df.iloc[start : start + chunk_size]
        images = [load_image(r) for _, r in rows.iterrows()]
        captions = rows[caption_col].tolist()

        t0 = time.perf_counter()
        js = judge.score(images, captions)
        t1 = time.perf_counter()
        cs = clip.score(images, captions)
        t2 = time.perf_counter()

        chunk = pd.DataFrame(
            {
                "judge": [s.expected for s in js],
                "judge_argmax": [s.argmax for s in js],
                "digit_mass": [s.digit_mass for s in js],
                **{f"p{d}": [s.probs[d - 1] for s in js] for d in range(1, 6)},
                "clipscore": cs,
            },
            index=rows.index,
        )
        chunk.to_parquet(path)
        chunks.append(chunk)
        timing["n"] += len(rows)
        timing["judge_s"] += t1 - t0
        timing["clip_s"] += t2 - t1
        timing["parse_failures"] += sum(not s.parse_ok for s in js)
    return pd.concat(chunks), timing


def throughput(n: int, seconds: float, usd_per_hour: float | None) -> dict:
    ips = n / seconds if seconds else float("nan")
    return {
        "images_per_s": ips,
        "usd_per_1k_images": (usd_per_hour / 3600 / ips * 1000) if usd_per_hour else None,
    }
