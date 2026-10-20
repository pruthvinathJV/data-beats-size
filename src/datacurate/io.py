"""Config, secrets, experiment tracking and artifact storage.

Importing this module loads .env, so import it before anything that reads
HF_HOME or other env vars at import time (huggingface_hub, transformers, vllm).
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

load_dotenv(ROOT / ".env")


def load_config(path: str | Path) -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def init_wandb(cfg: dict, job_type: str, **extra_config):
    import wandb

    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "data-beats-size"),
        name=f"{cfg['run_name']}-{job_type}",
        job_type=job_type,
        config={**cfg, **extra_config},
        mode="online" if os.getenv("WANDB_API_KEY") else "offline",
    )


def upload_to_s3(local: Path, prefix: str) -> str | None:
    """Mirror a file or directory to s3://$S3_BUCKET/<prefix>/. No-op if S3_BUCKET is unset."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return None
    import boto3

    s3 = boto3.client("s3")
    files = [local] if local.is_file() else sorted(p for p in local.rglob("*") if p.is_file())
    for f in files:
        s3.upload_file(str(f), bucket, f"{prefix}/{f.relative_to(local.parent)}")
    return f"s3://{bucket}/{prefix}"


def download_from_s3(prefix: str, dest: Path) -> int:
    """Mirror s3://$S3_BUCKET/<prefix>/ into dest/. Returns the number of files fetched."""
    import boto3

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            target = dest / obj["Key"][len(prefix) + 1 :]
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))
            n += 1
    return n
