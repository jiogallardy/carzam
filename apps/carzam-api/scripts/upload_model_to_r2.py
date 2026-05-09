"""One-time: upload the trained checkpoint + classes.json to R2.

The carzam-api container fetches these from R2 at startup. Re-run this
script whenever you train a new checkpoint and want to ship it.

Usage (from monorepo root):
    R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \\
    R2_BUCKET=carzam-clips \\
        uv run python apps/carzam-api/scripts/upload_model_to_r2.py \\
            runs/20260507_174723

That uploads:
    runs/20260507_174723/checkpoint.pt  -> s3://<bucket>/model/checkpoint.pt
    runs/20260507_174723/classes.json   -> s3://<bucket>/model/classes.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    ckpt = run_dir / "checkpoint.pt"
    classes = run_dir / "classes.json"
    for p in (ckpt, classes):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1

    account = os.environ["R2_ACCOUNT_ID"]
    key_id = os.environ["R2_ACCESS_KEY_ID"]
    secret = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.environ.get("R2_BUCKET", "carzam-clips")

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", region_name="auto"),
    )

    print(f"uploading {ckpt} -> s3://{bucket}/model/checkpoint.pt ({ckpt.stat().st_size / 1e6:.1f} MB)")
    client.upload_file(str(ckpt), bucket, "model/checkpoint.pt")
    print(f"uploading {classes} -> s3://{bucket}/model/classes.json")
    client.upload_file(str(classes), bucket, "model/classes.json")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
