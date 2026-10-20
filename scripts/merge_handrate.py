"""Merge per-batch rating CSVs into data/handrate/ratings.csv and report progress.

    python scripts/merge_handrate.py                       # reads data/handrate/ratings/
    python scripts/merge_handrate.py ~/Documents/ratings   # or any folders / CSV files you saved to

Re-downloads of a batch are fine (Chrome names them "... (1).csv"): for each
item, the most recent rating wins.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from datacurate.io import DATA, RESULTS, ROOT

d = DATA / "handrate"
items = pd.read_parquet(d / "items.parquet")
sources = [Path(a).expanduser() for a in sys.argv[1:]] or [d / "ratings"]
files = sorted(
    f for s in sources for f in ([s] if s.is_file() else s.iterdir())
    if f.is_file() and not f.name.startswith(".")
)
if not files:
    raise SystemExit(f"No rating CSVs found in {', '.join(map(str, sources))}.")

frames = [pd.read_csv(f, dtype={"key": str, "tags": str, "note": str}).assign(file=f.name) for f in files]
frames = [f for f in frames if {"key", "rating", "rated_at"} <= set(f.columns)]
r = pd.concat(frames).dropna(subset=["rating"])
r = r.sort_values("rated_at").drop_duplicates("key", keep="last")

unknown = set(r.key) - set(items.key)
if unknown:
    raise SystemExit(f"{len(unknown)} keys not in items.parquet, e.g. {sorted(unknown)[:3]}")
bad = r[~r.rating.isin([1, 2, 3, 4, 5])]
if len(bad):
    raise SystemExit(f"Invalid ratings: {bad[['key', 'rating']].to_dict('records')}")

r["rating"] = r.rating.astype(int)
out = items[["key"]].merge(r[["key", "rating", "tags", "note", "rated_at"]], on="key", how="left")
out.to_csv(d / "ratings.csv", index=False)

rated = out.rating.notna()
print(f"{rated.sum()} / {len(out)} rated, from {len(files)} file(s) -> {d / 'ratings.csv'}")
print("scores:", out.rating.dropna().astype(int).value_counts().sort_index().to_dict())
tags = out.tags.dropna().str.split(";").explode()
if len(tags):
    print("tags:  ", tags.value_counts().to_dict())
print("notes: ", out.note.notna().sum())
missing_batches = sorted({i // 15 + 1 for i in out.index[~rated]})
print("batches with unrated items:", missing_batches or "none")

# Fingerprint for the pre-registration: committed to git before any model scoring,
# so the exact labels are provable without publishing them.
manifest = {
    "ratings_csv_sha256": hashlib.sha256((d / "ratings.csv").read_bytes()).hexdigest(),
    "items_parquet_sha256": hashlib.sha256((d / "items.parquet").read_bytes()).hexdigest(),
    "rubric_sha256": hashlib.sha256((ROOT / "prompts/judge_rubric_v1.md").read_bytes()).hexdigest(),
    "n_items": len(out),
    "n_rated": int(rated.sum()),
    "score_counts": {int(k): int(v) for k, v in out.rating.dropna().astype(int).value_counts().sort_index().items()},
}
RESULTS.mkdir(exist_ok=True)
(RESULTS / "handrate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("fingerprint -> results/handrate_manifest.json")
