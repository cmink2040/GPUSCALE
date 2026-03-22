"""Upload model artifacts to Wasabi S3."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

# Wasabi S3-compatible endpoint. Region-specific endpoints also exist
# (e.g. s3.us-east-1.wasabisys.com) but the generic one routes automatically.
DEFAULT_WASABI_ENDPOINT = "https://s3.wasabisys.com"
DEFAULT_WASABI_REGION = "us-east-1"

# Files above this threshold use multipart upload (100 MB).
MULTIPART_THRESHOLD = 100 * 1024 * 1024
MULTIPART_CHUNKSIZE = 100 * 1024 * 1024


def get_s3_client(
    endpoint_url: str | None = None,
    region: str | None = None,
) -> boto3.client:
    """Create a boto3 S3 client configured for Wasabi.

    Credentials are read from environment variables:
        WASABI_ACCESS_KEY, WASABI_SECRET_KEY

    Args:
        endpoint_url: S3-compatible endpoint. Defaults to Wasabi's endpoint.
        region: AWS/Wasabi region. Defaults to us-east-1.

    Returns:
        A configured boto3 S3 client.

    Raises:
        EnvironmentError: If required credentials are missing.
    """
    access_key = os.environ.get("WASABI_ACCESS_KEY")
    secret_key = os.environ.get("WASABI_SECRET_KEY")

    if not access_key or not secret_key:
        raise EnvironmentError(
            "WASABI_ACCESS_KEY and WASABI_SECRET_KEY environment variables must be set. "
            "See .env.example in the project root."
        )

    endpoint = endpoint_url or os.environ.get("WASABI_ENDPOINT", DEFAULT_WASABI_ENDPOINT)
    rgn = region or os.environ.get("WASABI_REGION", DEFAULT_WASABI_REGION)

    log.debug("Creating S3 client: endpoint=%s region=%s", endpoint, rgn)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=rgn,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=10,
        ),
    )


def get_bucket_name() -> str:
    """Return the configured bucket name from environment.

    Raises:
        EnvironmentError: If WASABI_BUCKET is not set.
    """
    bucket = os.environ.get("WASABI_BUCKET")
    if not bucket:
        raise EnvironmentError("WASABI_BUCKET environment variable must be set.")
    return bucket


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads the file in 8 KB chunks to handle large model files without
    consuming excessive memory.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def upload_directory(
    local_dir: Path,
    s3_prefix: str,
    s3_client: boto3.client | None = None,
    bucket: str | None = None,
) -> list[dict]:
    """Upload all files in a local directory to S3 under the given prefix.

    Args:
        local_dir: Local directory containing files to upload.
        s3_prefix: The S3 key prefix (e.g. "meta-llama/Llama-3.1-8B-Instruct/full/").
        s3_client: Optional pre-configured S3 client.
        bucket: Bucket name override. Defaults to WASABI_BUCKET env var.

    Returns:
        List of dicts with upload metadata: {"key": ..., "size": ..., "sha256": ...}
    """
    client = s3_client or get_s3_client()
    bucket_name = bucket or get_bucket_name()

    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    uploaded: list[dict] = []
    files = sorted(f for f in local_dir.rglob("*") if f.is_file())

    if not files:
        log.warning("No files found in %s — nothing to upload", local_dir)
        return uploaded

    log.info("Uploading %d file(s) from %s to s3://%s/%s", len(files), local_dir, bucket_name, s3_prefix)

    for file_path in files:
        relative = file_path.relative_to(local_dir)
        s3_key = f"{s3_prefix.rstrip('/')}/{relative}"
        file_size = file_path.stat().st_size

        # Skip upload if the object already exists with the same size (cheap check).
        if _object_exists(client, bucket_name, s3_key, expected_size=file_size):
            log.info("Skipping %s (already exists with same size)", s3_key)
            uploaded.append({
                "key": s3_key,
                "size": file_size,
                "sha256": "",  # Skip hash for skipped files
                "skipped": True,
            })
            continue

        sha256 = compute_sha256(file_path)
        log.info("Uploading %s (%s bytes) -> s3://%s/%s", file_path.name, file_size, bucket_name, s3_key)

        extra_args = {"Metadata": {"sha256": sha256}}

        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
        )

        client.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            ExtraArgs=extra_args,
            Config=transfer_config,
        )

        log.info("Uploaded %s", s3_key)
        uploaded.append({
            "key": s3_key,
            "size": file_size,
            "sha256": sha256,
            "skipped": False,
        })

    return uploaded


def _object_exists(
    client: boto3.client,
    bucket: str,
    key: str,
    expected_size: int | None = None,
) -> bool:
    """Check if an S3 object exists, optionally matching expected size."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        if expected_size is not None:
            return resp["ContentLength"] == expected_size
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def list_bucket_objects(
    prefix: str = "",
    s3_client: boto3.client | None = None,
    bucket: str | None = None,
) -> list[dict]:
    """List all objects in the bucket under the given prefix.

    Returns a list of dicts: {"key": str, "size": int, "last_modified": datetime}.
    """
    client = s3_client or get_s3_client()
    bucket_name = bucket or get_bucket_name()

    objects: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")

    kwargs: dict = {"Bucket": bucket_name}
    if prefix:
        kwargs["Prefix"] = prefix

    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"],
            })

    return objects
