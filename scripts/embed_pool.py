"""Stage 2b: CLIP image + text embeddings and CLIPScore for the pool, via Ray Data.

Also embeds the Karpathy test images, used for decontamination.

    python scripts/embed_pool.py
"""

import io

import numpy as np
import pandas as pd
import ray
from PIL import Image

from datacurate.clipscore import CLIPSCORE_W, CLIPScorer
from datacurate.io import DATA, load_config

POOL = DATA / "pool"


class Embedder:
    def __init__(self, model: str):
        self.clip = CLIPScorer(model, batch_size=256)

    def __call__(self, batch: dict) -> dict:
        images = [Image.open(io.BytesIO(b)) for b in batch["img"]]
        img = self.clip.embed_images(images)
        out = {"key": batch["key"], "img_emb": list(img.astype(np.float16))}
        if "caption" in batch:
            txt = self.clip.embed_texts(list(batch["caption"]))
            out["txt_emb"] = list(txt.astype(np.float16))
            out["clipscore"] = CLIPSCORE_W * np.clip((img * txt).sum(1), 0, None)
        return out


def run(src, dst, model: str, columns: list[str]):
    ds = ray.data.read_parquet(str(src), columns=columns)
    ds = ds.map_batches(Embedder, fn_constructor_args=(model,), batch_size=1024, concurrency=1, num_gpus=1)
    df = ds.to_pandas()
    df.to_parquet(dst)
    return df


def main():
    cfg = load_config("configs/polaris_calibration.yaml")
    ray.init()
    pool = run(POOL / "raw", POOL / "embeddings.parquet", cfg["clip"]["model"], ["key", "caption", "img"])
    print(f"pool: {len(pool):,} embedded, CLIPScore mean {pool.clipscore.mean():.3f}")
    coco = pd.read_parquet(DATA / "coco" / "karpathy_test.parquet", columns=["cocoid", "img"])
    coco["key"] = coco.cocoid.astype(str)
    tmp = DATA / "coco" / "_embed_input.parquet"
    coco[["key", "img"]].to_parquet(tmp)
    test = run(tmp, DATA / "coco" / "test_embeddings.parquet", cfg["clip"]["model"], ["key", "img"])
    tmp.unlink()
    print(f"karpathy test: {len(test):,} embedded")


if __name__ == "__main__":
    main()
