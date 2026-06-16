"""
Content-Addressed Storage (CAS) and S3 Presigned URL management.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from web.config import get_settings

settings = get_settings()


def get_content_hash(data: bytes) -> str:
    """Generate 16-char SHA-256 hash for content-addressing."""
    return hashlib.sha256(data).hexdigest()[:16]


def get_s3_client():
    """Build boto3 S3 client using config settings."""
    if not settings.use_s3:
        return None

    # Custom endpoint url (e.g., local MinIO)
    s3_config = Config(signature_version="s3v4")
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=s3_config,
    )


async def s3_object_exists(client, bucket: str, key: str) -> bool:
    """Check if object exists in S3 bucket."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


async def store_file(file_bytes: bytes, filename: str, category: str) -> str:
    """
    Store file bytes under content-addressable path.
    If S3 is enabled, uploads to S3 bucket.
    Otherwise, saves to local upload directory.
    Returns: stored_path (local absolute path or S3 key).
    """
    content_hash = get_content_hash(file_bytes)
    ext = Path(filename).suffix

    # Separate categories into directories
    if category == "lob":
        local_dir = settings.lob_upload_dir
        s3_prefix = "lob"
    elif category == "agg_trades":
        local_dir = settings.agg_upload_dir
        s3_prefix = "agg_trades"
    else:
        local_dir = settings.model_upload_dir
        s3_prefix = "models"

    s3_key = f"{s3_prefix}/{content_hash}{ext}"

    if settings.use_s3:
        client = get_s3_client()
        if client:
            if not await s3_object_exists(client, settings.s3_bucket_name, s3_key):
                client.put_object(
                    Bucket=settings.s3_bucket_name,
                    Key=s3_key,
                    Body=file_bytes,
                )
            return f"s3://{settings.s3_bucket_name}/{s3_key}"

    # Local fallback
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{content_hash}_{filename}"
    if not local_path.exists():
        with open(local_path, "wb") as f:
            f.write(file_bytes)
    return str(local_path.resolve())


def get_presigned_upload_url(filename: str, category: str) -> dict[str, Any]:
    """
    Generate S3 presigned POST dictionary for direct browser upload.
    Falls back to a local upload endpoint if S3 is disabled.
    """
    ext = Path(filename).suffix
    # Category routing
    s3_prefix = "lob" if category == "lob" else ("agg_trades" if category == "agg_trades" else "models")

    # We use a placeholder hash or filename as prefix to avoid collisions
    random_prefix = hashlib.md5(filename.encode()).hexdigest()[:8]
    s3_key = f"{s3_prefix}/{random_prefix}_{filename}"

    if settings.use_s3:
        client = get_s3_client()
        if client:
            try:
                response = client.generate_presigned_post(
                    Bucket=settings.s3_bucket_name,
                    Key=s3_key,
                    Fields=None,
                    Conditions=None,
                    ExpiresIn=3600,
                )
                return {
                    "url": response["url"],
                    "fields": response["fields"],
                    "s3_key": s3_key,
                    "use_s3": True,
                }
            except Exception as e:
                # Log error and fall back to local
                pass

    # Local upload fallback descriptor
    return {
        "url": f"/api/upload/local?category={category}",
        "fields": {},
        "s3_key": None,
        "use_s3": False,
    }
