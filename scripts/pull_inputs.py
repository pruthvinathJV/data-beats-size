"""On the pod: fetch the hand-rating set from S3 and check it against the committed fingerprint.

    python scripts/pull_inputs.py
"""

import hashlib
import json

from datacurate.io import DATA, RESULTS, download_from_s3

n = download_from_s3("inputs/handrate", DATA / "handrate")
manifest = json.loads((RESULTS / "handrate_manifest.json").read_text())
for name, key in [("ratings.csv", "ratings_csv_sha256"), ("items.parquet", "items_parquet_sha256")]:
    digest = hashlib.sha256((DATA / "handrate" / name).read_bytes()).hexdigest()
    if digest != manifest[key]:
        raise SystemExit(f"{name} does not match the committed fingerprint. Stop and investigate.")
print(f"{n} files -> {DATA / 'handrate'}; ratings and items match the committed fingerprint")
