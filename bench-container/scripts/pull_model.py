#!/usr/bin/env python3
"""Pull model into MODEL_DIR — checks persistent volume first, then S3, then HuggingFace."""

import os
import sys

MODEL = os.environ.get("MODEL", "")
MODEL_FORMAT = os.environ.get("MODEL_FORMAT", "")
GGUF_QUANT = os.environ.get("GGUF_QUANT", "")
ENGINE = os.environ.get("ENGINE", "")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "")

# MODEL_DIR set by entrypoint.sh — uses persistent volume if available
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")


def already_downloaded() -> bool:
    """Check if model files already exist (from a previous run on persistent storage)."""
    if not os.path.isdir(MODEL_DIR):
        return False
    files = [f for f in os.listdir(MODEL_DIR) if not f.startswith(".") and os.path.getsize(os.path.join(MODEL_DIR, f)) > 0]
    if not files:
        return False
    # Check for meaningful files (not just small metadata)
    large_files = [f for f in files if os.path.getsize(os.path.join(MODEL_DIR, f)) > 1_000_000]
    if large_files:
        total_mb = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in files) / 1e6
        print(f"Model already present in {MODEL_DIR} ({len(files)} files, {total_mb:.0f} MB). Skipping download.", file=sys.stderr)
        return True
    return False


def pull_from_s3():
    """Download model files from Wasabi S3."""
    import boto3

    endpoint = os.environ["S3_ENDPOINT"]
    bucket = os.environ["S3_BUCKET"]
    key_prefix = S3_MODEL_KEY.rstrip("/")

    print(f"Pulling from S3: s3://{bucket}/{key_prefix}/", file=sys.stderr)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(key_prefix):].lstrip("/")
            if not relative:
                continue

            dest = os.path.join(MODEL_DIR, relative)
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            # Skip if already exists and same size
            if os.path.exists(dest) and os.path.getsize(dest) == obj["Size"]:
                print(f"  Already exists: {relative}", file=sys.stderr)
                skipped += 1
                continue

            size_mb = obj["Size"] / 1e6
            print(f"  Downloading {relative} ({size_mb:.1f} MB)...", file=sys.stderr)
            client.download_file(bucket, key, dest)
            downloaded += 1

    print(f"S3 download complete. {downloaded} downloaded, {skipped} skipped.", file=sys.stderr)


def pull_from_huggingface():
    """Download model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN")

    if MODEL_FORMAT == "gguf" and GGUF_QUANT:
        repo_id = os.environ.get("HF_REPO_ID", MODEL)
        filename = f"{GGUF_QUANT}.gguf"
        print(f"Pulling GGUF from HF: {repo_id}/{filename}", file=sys.stderr)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODEL_DIR,
            token=token,
        )
    else:
        print(f"Pulling from HF: {MODEL}", file=sys.stderr)
        snapshot_download(
            repo_id=MODEL,
            local_dir=MODEL_DIR,
            token=token,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )

    print("HuggingFace download complete.", file=sys.stderr)


def main():
    if not MODEL:
        print("ERROR: MODEL env var is required", file=sys.stderr)
        sys.exit(1)

    # Check if model already exists on persistent storage
    if already_downloaded():
        return

    # Priority 1: Pull from S3 (private/gated models — includes Meta .pth format)
    if os.environ.get("S3_BUCKET") and S3_MODEL_KEY:
        pull_from_s3()
        return

    # Priority 2: Pull from HuggingFace (public models)
    pull_from_huggingface()


if __name__ == "__main__":
    main()
