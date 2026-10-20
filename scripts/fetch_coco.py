"""Fetch the evaluation set: COCO Karpathy test (5K images, 5 references each),
plus what CHAIR needs (val2014 instance annotations and the CHAIR synonym list).

    python scripts/fetch_coco.py
"""

import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from datacurate.io import DATA

OUT = DATA / "coco"
SYNONYMS_URL = "https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
    raise AssertionError


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(
        hf_hub_download("yerevann/coco-karpathy", "data/test-00000-of-00001.parquet", repo_type="dataset")
    )
    test["refs"] = test.sentences.map(lambda s: [x.strip() for x in s])
    if not (OUT / "karpathy_test.parquet").exists():
        with ThreadPoolExecutor(64) as pool:
            test["img"] = list(pool.map(get, test.url))
        test[["cocoid", "filename", "url", "refs", "img"]].to_parquet(OUT / "karpathy_test.parquet")
    print(f"{len(test)} test images -> {OUT / 'karpathy_test.parquet'}")

    (OUT / "chair_synonyms.txt").write_bytes(get(SYNONYMS_URL))
    inst = OUT / "instances_val2014.json"
    if not inst.exists():
        with zipfile.ZipFile(io.BytesIO(get(ANNOTATIONS_URL))) as z:
            inst.write_bytes(z.read("annotations/instances_val2014.json"))
    n = len(json.loads(inst.read_text())["annotations"])
    print(f"CHAIR inputs ready: synonyms + {n:,} val2014 instance annotations")


if __name__ == "__main__":
    main()
