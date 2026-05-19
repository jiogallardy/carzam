"""Upload a trained checkpoint + classes.json to R2.

The carzam-api container fetches these on startup. Re-run whenever you
ship a new checkpoint.

Two modes:

    # Versioned slot — shows up in the mobile picker as `v7`:
    R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \\
        uv run python apps/carzam-api/scripts/upload_model_to_r2.py \\
            runs/20260509_024915 --id v7

    # Legacy default slot (model/checkpoint.pt) — what the API treats
    # as id "default" if no DEFAULT_MODEL_ID env is set:
    R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \\
        uv run python apps/carzam-api/scripts/upload_model_to_r2.py \\
            runs/20260509_024915

You can also do both — pass --id and the script writes to BOTH locations
so the new checkpoint is the versioned option AND the default.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def upload_pair(client, bucket: str, ckpt: Path, classes: Path, ckpt_key: str, classes_key: str) -> None:
    print(f"  -> s3://{bucket}/{ckpt_key} ({ckpt.stat().st_size / 1e6:.1f} MB)")
    client.upload_file(str(ckpt), bucket, ckpt_key)
    print(f"  -> s3://{bucket}/{classes_key}")
    client.upload_file(str(classes), bucket, classes_key)


def upload_optional_extras(
    client, bucket: str, run_dir: Path, key_prefix: str,
) -> None:
    """Upload any extra artifacts the run dir might contain (prototypes for
    contrastive runs, config.yaml). Each is optional — silently skipped if
    not present so this is safe for non-contrastive runs."""
    for filename in ("prototypes.pt", "config.yaml"):
        path = run_dir / filename
        if not path.exists():
            continue
        key = f"{key_prefix}{filename}"
        size_mb = path.stat().st_size / 1e6
        print(f"  -> s3://{bucket}/{key} ({size_mb:.1f} MB)")
        client.upload_file(str(path), bucket, key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="path to the training run dir containing checkpoint.pt + classes.json")
    parser.add_argument(
        "--id",
        dest="model_id",
        default=None,
        help="versioned id, e.g. v7 or 20260509_024915. If set, uploads to model/<id>/.",
    )
    parser.add_argument(
        "--also-default",
        action="store_true",
        help="When --id is set, ALSO upload to the legacy model/checkpoint.pt path so this checkpoint is the default.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
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

    if args.model_id:
        print(f"uploading {run_dir.name} as version '{args.model_id}'")
        upload_pair(
            client, bucket, ckpt, classes,
            f"model/{args.model_id}/checkpoint.pt",
            f"model/{args.model_id}/classes.json",
        )
        upload_optional_extras(client, bucket, run_dir, f"model/{args.model_id}/")
        if args.also_default:
            print("also writing to legacy default slot")
            upload_pair(client, bucket, ckpt, classes, "model/checkpoint.pt", "model/classes.json")
            upload_optional_extras(client, bucket, run_dir, "model/")
    else:
        print(f"uploading {run_dir.name} to legacy default slot (model/checkpoint.pt)")
        upload_pair(client, bucket, ckpt, classes, "model/checkpoint.pt", "model/classes.json")
        upload_optional_extras(client, bucket, run_dir, "model/")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
