"""Stage 2c: raw pool -> B pool. Heuristics, approved agent rules, near-duplicate
removal, decontamination against the Karpathy test images.

Writes data/pool/pool.parquet (every raw row, with `removed_by`) and logs the
count removed at each step to results/pool_filtering.json.

    python scripts/filter_pool.py
"""

import json

import numpy as np
import pandas as pd
import yaml

from datacurate.filters import add_features, apply_rules, max_cosine
from datacurate.io import DATA, RESULTS, ROOT

POOL = DATA / "pool"


def main():
    cfg = yaml.safe_load((ROOT / "configs/filters.yaml").read_text())
    meta_cols = ["key", "shard", "caption", "url", "width", "height", "gray_std", "n_words"]
    raw = pd.read_parquet(POOL / "raw", columns=meta_cols)
    emb = pd.read_parquet(POOL / "embeddings.parquet")
    df = add_features(raw.merge(emb[["key", "clipscore"]], on="key"))
    log = {"raw": len(df)}

    rules = {**cfg["heuristics"], **{r["name"]: r for r in (cfg.get("agent_rules") or [])}}
    df["removed_by"] = apply_rules(df, rules)
    for name in rules:
        log[f"rule:{name}"] = int((df.removed_by == name).sum())

    # Embeddings aligned to df order.
    e = emb.set_index("key").loc[df.key]
    img = np.stack(e.img_emb.to_numpy())

    keep = (df.removed_by == "").to_numpy()
    idx = np.flatnonzero(keep)
    dup = max_cosine(img[idx], img[idx], exclude_earlier=True) >= cfg["dedup"]["image_cosine"]
    df.loc[df.index[idx[dup]], "removed_by"] = "near_duplicate"
    log["near_duplicate"] = int(dup.sum())

    test = pd.read_parquet(DATA / "coco" / "test_embeddings.parquet")
    test_emb = np.stack(test.img_emb.to_numpy())
    idx = np.flatnonzero((df.removed_by == "").to_numpy())
    contam = max_cosine(img[idx], test_emb) >= cfg["decontamination"]["image_cosine"]
    df.loc[df.index[idx[contam]], "removed_by"] = "decontamination"
    log["decontamination"] = int(contam.sum())

    log["b_pool"] = int((df.removed_by == "").sum())
    log["removed_share"] = round(1 - log["b_pool"] / log["raw"], 4)
    df.drop(columns=["min_side", "aspect"]).to_parquet(POOL / "pool.parquet")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pool_filtering.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
