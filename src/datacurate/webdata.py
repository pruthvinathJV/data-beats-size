"""Streaming reader for CC12M WebDataset shards on the Hugging Face Hub."""

import os
import tarfile
from collections.abc import Iterator

import requests

CC12M_URL = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{:04d}.tar"


def iter_samples(url: str, scan_limit: int | None = None) -> Iterator[tuple[str, dict[str, bytes]]]:
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
