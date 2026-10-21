"""Verify every key in .env works, without printing any secret.

    python scripts/check_env.py
"""

import os
import uuid

import requests

from datacurate.io import ROOT  # loads .env


def check_hf():
    from huggingface_hub import whoami

    return f"logged in as {whoami(token=os.environ['HF_TOKEN'])['name']}"


def check_wandb():
    r = requests.post(
        "https://api.wandb.ai/graphql",
        json={"query": "{ viewer { username } }"},
        auth=("api", os.environ["WANDB_API_KEY"]),
        timeout=20,
    )
    r.raise_for_status()
    return f"logged in as {r.json()['data']['viewer']['username']}"


def check_s3():
    import boto3

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    key = f"_healthcheck/{uuid.uuid4().hex}"
    s3.put_object(Bucket=bucket, Key=key, Body=b"ok")
    assert s3.get_object(Bucket=bucket, Key=key)["Body"].read() == b"ok"
    s3.delete_object(Bucket=bucket, Key=key)
    return f"write/read/delete ok on s3://{bucket}"


CHECKS = {
    "HF_TOKEN": check_hf,
    "WANDB_API_KEY": check_wandb,
    "S3_BUCKET": check_s3,
}
REQUIRED = {"HF_TOKEN"}  # W&B and S3 are optional

if __name__ == "__main__":
    if not (ROOT / ".env").exists():
        raise SystemExit("No .env yet: cp .env.example .env and fill it in.")
    failed = False
    for var, fn in CHECKS.items():
        if not os.getenv(var):
            print(f"  --  {var}: not set" + ("" if var in REQUIRED else " (optional)"))
            failed |= var in REQUIRED
            continue
        try:
            print(f"  ok  {var}: {fn()}")
        except Exception as e:  # noqa: BLE001 - report and keep checking the rest
            print(f"FAIL  {var}: {type(e).__name__}: {str(e)[:160]}")
            failed = True
    raise SystemExit(1 if failed else 0)
