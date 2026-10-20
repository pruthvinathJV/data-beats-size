"""Polaris (yuwd/Polaris): human judgments of caption quality on COCO / nocaps images.

Each row is ONE rater's judgment, with human_score in {0, .25, .5, .75, 1}.
Some (image, caption) pairs were rated more than once; we collapse to unique
pairs (mean rating) and keep the individual ratings for the human ceiling.
There are no image IDs, so images are keyed by a hash of their bytes.
"""

import hashlib

import numpy as np
import pandas as pd

REPO = "yuwd/Polaris"


def to_five_point(score: pd.Series) -> pd.Series:
    return 1 + 4 * score


def shard_files(split: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(
        f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith(f"data/{split}-")
    )


def load_polaris(split: str, n_shards: int | None = None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    frames = []
    for f in shard_files(split)[:n_shards]:
        d = pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset"))
        frames.append(
            pd.DataFrame(
                {
                    "img_bytes": d.img.map(lambda x: x["bytes"]),
                    "cand": d.cand,
                    "refs": d.refs.map(list),
                    "human": to_five_point(d.human_score.astype(float)),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["img_hash"] = df.img_bytes.map(lambda b: hashlib.sha1(b).hexdigest())
    return df


def collapse_ratings(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["img_hash", "cand"], sort=False)
        .agg(
            img_bytes=("img_bytes", "first"),
            refs=("refs", "first"),
            human=("human", "mean"),
            n_ratings=("human", "size"),
            ratings=("human", list),
        )
        .reset_index()
    )
    out["n_words"] = out.cand.str.split().str.len()
    return out


def rater_pairs(pairs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """For pairs with 2+ ratings, draw two distinct ratings: rater A vs rater B."""
    rng = np.random.default_rng(seed)
    multi = pairs[pairs.n_ratings >= 2].copy()
    picks = np.array([rng.choice(np.asarray(r), 2, replace=False) for r in multi.ratings])
    multi["r1"], multi["r2"] = picks[:, 0], picks[:, 1]
    return multi
