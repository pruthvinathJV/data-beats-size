"""Stage 2a: download the raw CC12M pool with Ray, one task per shard.

Each shard becomes one parquet file with a resized JPEG and cheap features used
by the heuristic filters (image spread for near-blank images, caption stats).

    python scripts/build_pool.py --shards 0 20
"""

import argparse
import io
import json
import time

import numpy as np
import pandas as pd
import ray
from PIL import Image

from datacurate.io import DATA
from datacurate.webdata import CC12M_URL, iter_samples

OUT = DATA / "pool" / "raw"
MAX_SIDE = 448


def jpeg_bytes(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@ray.remote(num_cpus=2)
def fetch_shard(shard: int) -> dict:
    path = OUT / f"shard_{shard:04d}.parquet"
    if path.exists():
        return {"shard": shard, "rows": len(pd.read_parquet(path, columns=["key"])), "cached": True}
    rows, bad = [], 0
    for key, s in iter_samples(CC12M_URL.format(shard)):
        if "jpg" not in s or "txt" not in s:
            bad += 1
            continue
        try:
            img = Image.open(io.BytesIO(s["jpg"]))
            img.load()
        except Exception:  # noqa: BLE001 - corrupt image: count and skip
            bad += 1
            continue
        meta = json.loads(s.get("json", b"{}"))
        caption = s["txt"].decode("utf-8", "replace").strip()
        gray = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
        rows.append({
            "key": key,
            "shard": shard,
            "caption": caption,
            "url": meta.get("url"),
            "width": img.width,
            "height": img.height,
            "gray_std": float(gray.std()),  # ~0 for blank or single-color images
            "n_words": len(caption.split()),
            "img": jpeg_bytes(img),
        })
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    tmp.rename(path)
    return {"shard": shard, "rows": len(rows), "bad": bad, "cached": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs=2, type=int, default=[0, 20], metavar=("START", "STOP"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ray.init()
    t0 = time.time()
    results = ray.get([fetch_shard.remote(s) for s in range(*args.shards)])
    total = sum(r["rows"] for r in results)
    print(json.dumps(results, indent=1))
    print(f"{total:,} pairs in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
