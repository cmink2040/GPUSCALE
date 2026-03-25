#!/usr/bin/env python3
"""Pull model from Wasabi S3 or HuggingFace Hub into /models/."""

import os
import sys

MODEL = os.environ.get("MODEL", "")
MODEL_FORMAT = os.environ.get("MODEL_FORMAT", "")
GGUF_QUANT = os.environ.get("GGUF_QUANT", "")
MODEL_DIR = "/models"


def pull_from_s3():
    """Download model files from Wasabi S3."""
    import boto3

    endpoint = os.environ["S3_ENDPOINT"]
    bucket = os.environ["S3_BUCKET"]
    key_prefix = os.environ["S3_MODEL_KEY"].rstrip("/")

    print(f"Pulling from S3: s3://{bucket}/{key_prefix}/", file=sys.stderr)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Get relative path after the prefix
            relative = key[len(key_prefix) :].lstrip("/")
            if not relative:
                continue

            dest = os.path.join(MODEL_DIR, relative)
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            size_mb = obj["Size"] / 1e6
            print(f"  Downloading {relative} ({size_mb:.1f} MB)...", file=sys.stderr)
            client.download_file(bucket, key, dest)

    print("S3 download complete.", file=sys.stderr)


def pull_from_huggingface():
    """Download model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN")

    if MODEL_FORMAT == "gguf" and GGUF_QUANT:
        # For GGUF, download only the specific quant file
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
        # Download the full repo
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

    # Determine source: S3 if credentials are present, otherwise HuggingFace
    if os.environ.get("S3_BUCKET") and os.environ.get("S3_MODEL_KEY"):
        pull_from_s3()
    else:
        pull_from_huggingface()


if __name__ == "__main__":
    main()
