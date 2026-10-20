"""Build the blind hand-rating set: 200 CC12M pairs from a held-out shard.

Runs on a laptop, no GPU. Run it, and rate everything, BEFORE any judge or
CLIPScore output exists for these pairs. score_handrate.py refuses to run
until your ratings file is in place.

    python scripts/make_handrate_set.py
    open data/handrate/pages/batch_01.html   # CSV auto-downloads when the page is done
    mv ~/Downloads/handrate_batch_*.csv data/handrate/ratings/
    python scripts/merge_handrate.py          # -> data/handrate/ratings.csv + progress

The pool for the experiment uses the first shards (0000-0039); this set comes
from the last shard, so it never overlaps the training pool.
"""

import argparse
import base64
import hashlib
import heapq
import io
import json
import os
import tarfile
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from datacurate.io import DATA, ROOT  # loads .env first
from datacurate.judge import resize_max_side

SHARD = "cc12m-train-2175.tar"
URL = f"https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/{SHARD}"


def iter_samples(url: str, scan_limit: int | None):
    """Stream a WebDataset tar, yielding (key, {ext: bytes}) per sample."""
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.getenv("HF_TOKEN") else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            key, sample, seen = None, {}, 0
            for m in tf:
                if not m.isfile():
                    continue
                k, ext = m.name.rsplit(".", 1)
                if key is not None and k != key:
                    yield key, sample
                    sample, seen = {}, seen + 1
                    if scan_limit and seen >= scan_limit:
                        return
                key = k
                sample[ext] = tf.extractfile(m).read()
            if sample:
                yield key, sample


def jpeg(img: Image.Image, side: int, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    resize_max_side(img, side).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-limit", type=int, default=None, help="stop early (for testing)")
    ap.add_argument("--page-only", action="store_true", help="re-render the rating pages from saved items")
    args = ap.parse_args()
    if args.page_only:
        return write_page(DATA / "handrate")

    # Seeded hash order = a uniform random sample, kept in a bounded heap while streaming.
    heap = []
    for key, s in iter_samples(URL, args.scan_limit):
        if "jpg" not in s or "txt" not in s:
            continue
        h = int(hashlib.sha1(f"{args.seed}:{key}".encode()).hexdigest(), 16)
        item = (-h, key, s)
        if len(heap) < args.n:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    picked = sorted(heap, reverse=True)  # ascending hash = presentation order

    out = DATA / "handrate"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, key, s in picked:
        img = Image.open(io.BytesIO(s["jpg"]))
        meta = json.loads(s.get("json", b"{}"))
        (out / "images" / f"{key}.jpg").write_bytes(jpeg(img, 768))
        rows.append({"key": key, "caption": s["txt"].decode("utf-8", "replace").strip(),
                     "width": img.width, "height": img.height, "url": meta.get("url"), "shard": SHARD})
    pd.DataFrame(rows).to_parquet(out / "items.parquet")
    write_page(out)


def write_page(out: Path, batch_size: int = 15):
    """Render one rating page per batch from items.parquet + images/ (never reads model output).

    Small batches mean a closed browser loses at most one batch of work.
    """
    items = pd.read_parquet(out / "items.parquet")
    page_items = [
        {"key": r.key, "caption": r.caption,
         "img": "data:image/jpeg;base64,"
                + base64.b64encode(jpeg(Image.open(out / "images" / f"{r.key}.jpg"), 512, 80)).decode()}
        for r in items.itertuples()
    ]
    rubric = (ROOT / "prompts/judge_rubric_v1.md").read_text().strip()
    template = (ROOT / "scripts/handrate_template.html").read_text()

    def js(x):  # safe inside <script>: a caption containing "</script>" must not end the tag
        return json.dumps(x).replace("</", "<\\/")

    pages = out / "pages"
    pages.mkdir(exist_ok=True)
    (out / "ratings").mkdir(exist_ok=True)
    n_batches = -(-len(page_items) // batch_size)
    for b in range(n_batches):
        batch = f"batch_{b + 1:02d}"
        chunk = page_items[b * batch_size : (b + 1) * batch_size]
        page = (
            template.replace("__ITEMS__", js(chunk))
            .replace("__RUBRIC__", js(rubric))
            .replace("__BATCH__", js(batch))
            .replace("__BATCH_LABEL__", f"{b + 1} of {n_batches}")
        )
        (pages / f"{batch}.html").write_text(page)
    (out / "rate.html").unlink(missing_ok=True)  # replaced by pages/
    print(f"{len(page_items)} items in {n_batches} pages -> {pages}")

if __name__ == "__main__":
    main()
