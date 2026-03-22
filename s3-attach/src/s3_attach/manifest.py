"""Generate and manage the manifest.json in the S3 bucket.

The manifest is a JSON index of all available models, their formats,
and checksums. It lives at the root of the bucket as manifest.json.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

from s3_attach.uploader import get_bucket_name, get_s3_client, list_bucket_objects

log = logging.getLogger(__name__)

MANIFEST_KEY = "manifest.json"


def build_manifest(
    s3_client: boto3.client | None = None,
    bucket: str | None = None,
) -> dict:
    """Scan the bucket and build a manifest dict describing all available models.

    The manifest structure:

    {
        "generated_at": "2025-01-15T12:00:00Z",
        "bucket": "gpuscale-models",
        "models": {
            "meta-llama/Llama-3.1-8B-Instruct": {
                "formats": {
                    "full": {
                        "files": [
                            {"key": "meta-llama/.../config.json", "size": 1234}
                        ]
                    },
                    "gguf": {
                        "files": [
                            {"key": "meta-llama/.../gguf/Q4_K_M.gguf", "size": 456789, "quant": "Q4_K_M"}
                        ]
                    },
                    "gptq": {
                        "4bit-128g": {
                            "files": [...]
                        }
                    }
                }
            }
        }
    }

    Returns:
        The manifest dict.
    """
    client = s3_client or get_s3_client()
    bucket_name = bucket or get_bucket_name()

    log.info("Scanning bucket s3://%s to build manifest", bucket_name)
    all_objects = list_bucket_objects(s3_client=client, bucket=bucket_name)

    # Filter out the manifest itself
    all_objects = [o for o in all_objects if o["key"] != MANIFEST_KEY]

    models: dict = {}

    for obj in all_objects:
        key = obj["key"]
        parts = key.split("/")

        # Expected structure: org/model/format_type/... or org/model/format_type/variant/...
        if len(parts) < 3:
            log.debug("Skipping object with unexpected path depth: %s", key)
            continue

        org = parts[0]
        model_name = parts[1]
        repo_id = f"{org}/{model_name}"
        format_type = parts[2]

        if repo_id not in models:
            models[repo_id] = {"formats": {}}

        file_entry = {
            "key": key,
            "size": obj["size"],
            "last_modified": obj["last_modified"].isoformat(),
        }

        if format_type == "gguf":
            if "gguf" not in models[repo_id]["formats"]:
                models[repo_id]["formats"]["gguf"] = {"files": []}
            # Extract quant name from filename if it ends with .gguf
            filename = parts[-1]
            if filename.endswith(".gguf"):
                file_entry["quant"] = filename.replace(".gguf", "")
            models[repo_id]["formats"]["gguf"]["files"].append(file_entry)

        elif format_type == "gptq":
            if "gptq" not in models[repo_id]["formats"]:
                models[repo_id]["formats"]["gptq"] = {}
            # GPTQ has a variant sub-directory (e.g. 4bit-128g)
            variant = parts[3] if len(parts) > 3 else "default"
            if variant not in models[repo_id]["formats"]["gptq"]:
                models[repo_id]["formats"]["gptq"][variant] = {"files": []}
            models[repo_id]["formats"]["gptq"][variant]["files"].append(file_entry)

        elif format_type == "full":
            if "full" not in models[repo_id]["formats"]:
                models[repo_id]["formats"]["full"] = {"files": []}
            models[repo_id]["formats"]["full"]["files"].append(file_entry)

        else:
            # Unknown format — still record it
            if format_type not in models[repo_id]["formats"]:
                models[repo_id]["formats"][format_type] = {"files": []}
            models[repo_id]["formats"][format_type]["files"].append(file_entry)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket_name,
        "models": models,
    }

    total_models = len(models)
    total_files = sum(1 for _ in all_objects)
    log.info("Manifest built: %d model(s), %d file(s)", total_models, total_files)

    return manifest


def upload_manifest(
    manifest: dict,
    s3_client: boto3.client | None = None,
    bucket: str | None = None,
) -> None:
    """Serialize the manifest dict to JSON and upload it to the bucket root.

    Args:
        manifest: The manifest dict (from build_manifest).
        s3_client: Optional pre-configured S3 client.
        bucket: Bucket name override.
    """
    client = s3_client or get_s3_client()
    bucket_name = bucket or get_bucket_name()

    manifest_json = json.dumps(manifest, indent=2, default=str)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(manifest_json)
        tmp_path = Path(tmp.name)

    try:
        log.info("Uploading manifest.json to s3://%s/%s", bucket_name, MANIFEST_KEY)
        client.upload_file(
            str(tmp_path),
            bucket_name,
            MANIFEST_KEY,
            ExtraArgs={"ContentType": "application/json"},
        )
        log.info("Manifest uploaded successfully")
    finally:
        tmp_path.unlink(missing_ok=True)


def regenerate_manifest(
    s3_client: boto3.client | None = None,
    bucket: str | None = None,
) -> dict:
    """Convenience function: build the manifest from bucket contents and upload it.

    Returns:
        The generated manifest dict.
    """
    client = s3_client or get_s3_client()
    bucket_name = bucket or get_bucket_name()

    manifest = build_manifest(s3_client=client, bucket=bucket_name)
    upload_manifest(manifest, s3_client=client, bucket=bucket_name)
    return manifest


def fetch_manifest(
    s3_client: boto3.client | None = None,
    bucket: str | None = None,
) -> dict | None:
    """Download and parse the current manifest.json from the bucket.

    Returns None if the manifest does not exist.
    """
    client = s3_client or get_s3_client()
    bucket_name = bucket or get_bucket_name()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        client.download_file(bucket_name, MANIFEST_KEY, str(tmp_path))
        with open(tmp_path) as f:
            return json.load(f)
    except client.exceptions.NoSuchKey:
        log.warning("No manifest.json found in bucket %s", bucket_name)
        return None
    except Exception as e:
        log.warning("Could not fetch manifest: %s", e)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)
