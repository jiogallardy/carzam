"""Upload one representative thumbnail per car class to R2.

Picks the largest *_0.jpg in data/visual_audit_cache/thumbnails/<car>/
(the YouTube maxresdefault) and writes it to R2 at car_images/<car>.jpg.
The carzam-api /car-classes/{id}/image route proxies these.

Usage:
    R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \\
        .venv/bin/python apps/carzam-api/scripts/upload_car_images.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from PIL import Image


def best_thumbnail(car_dir: Path) -> Path | None:
    """Return the largest *.jpg in `car_dir`, preferring `_0.jpg` (maxres)."""
    candidates = list(car_dir.glob("*_0.jpg"))
    if not candidates:
        candidates = list(car_dir.glob("*.jpg"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def downscale_for_mobile(src: Path, max_width: int = 800, jpeg_quality: int = 82) -> bytes:
    """Downscale to mobile-friendly size + recompress to keep R2 storage tight."""
    img = Image.open(src).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs-dir", type=Path,
                    default=Path("data/visual_audit_cache/thumbnails"))
    ap.add_argument("--prefix", default="car_images",
                    help="R2 key prefix where images get written.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.thumbs_dir.exists():
        print(f"missing: {args.thumbs_dir}", file=sys.stderr)
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

    car_dirs = sorted(p for p in args.thumbs_dir.iterdir() if p.is_dir())
    print(f"found {len(car_dirs)} car dirs under {args.thumbs_dir}")

    uploaded: dict[str, int] = {}
    skipped: list[str] = []
    for car_dir in car_dirs:
        car = car_dir.name
        thumb = best_thumbnail(car_dir)
        if thumb is None:
            skipped.append(car)
            print(f"  [skip] {car}: no thumbnails")
            continue
        body = downscale_for_mobile(thumb)
        key = f"{args.prefix}/{car}.jpg"
        if args.dry_run:
            print(f"  [dry] {car}  <- {thumb.name}  -> s3://{bucket}/{key} ({len(body)/1024:.0f} KB)")
        else:
            client.put_object(
                Bucket=bucket, Key=key, Body=body,
                ContentType="image/jpeg",
                CacheControl="public, max-age=2592000",  # 30 days
            )
            print(f"  [ok]  {car}  <- {thumb.name}  -> s3://{bucket}/{key} ({len(body)/1024:.0f} KB)")
        uploaded[car] = len(body)

    print(f"\nuploaded {len(uploaded)} images, skipped {len(skipped)}")
    if skipped:
        print(f"skipped cars (no thumbnails): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
