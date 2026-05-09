"""R2 (S3-compatible) client. We never expose raw R2 to the mobile app — uploads pass through this API."""
from __future__ import annotations

import boto3
from botocore.config import Config

from app.config import settings


def r2_client():
    cfg = settings()
    return boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint,
        aws_access_key_id=cfg.r2_access_key_id,
        aws_secret_access_key=cfg.r2_secret_access_key,
        config=Config(signature_version="s3v4", region_name="auto"),
    )


def put_clip(key: str, body: bytes, content_type: str = "audio/wav") -> None:
    cfg = settings()
    r2_client().put_object(
        Bucket=cfg.r2_bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def get_clip(key: str) -> bytes:
    cfg = settings()
    obj = r2_client().get_object(Bucket=cfg.r2_bucket, Key=key)
    return obj["Body"].read()


def download_to_path(key: str, dest: str) -> None:
    """Download an R2 object to a local path. Used for fetching the model checkpoint at boot."""
    cfg = settings()
    r2_client().download_file(cfg.r2_bucket, key, dest)


def presigned_download_url(key: str, expires_seconds: int = 3600) -> str:
    cfg = settings()
    return r2_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": cfg.r2_bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )
