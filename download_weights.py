"""
Dynamic Google Cloud Storage (GCS) Model Weight Downloader for RetroChimera.

Extracts model weights checkpoint dynamically from AIP_STORAGE_URI at container startup.
Decouples heavy model weights from Docker image layers for fast, scale-to-zero container boots.
"""

import logging
import os
import sys

logger = logging.getLogger("retrochimera-downloader")


def download_model_weights() -> None:
    """Download model weight checkpoint from GCS if AIP_STORAGE_URI is set."""
    storage_uri = os.environ.get("AIP_STORAGE_URI")
    logger.info("AIP_STORAGE_URI: %s", storage_uri)

    model_path = os.environ.get("RETROCHIMERA_MODEL_PATH", "checkpoint.pt")
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)
        os.environ["RETROCHIMERA_MODEL_PATH"] = model_path

    if not storage_uri:
        logger.warning("AIP_STORAGE_URI is not set. Skipping cloud weight download.")
        return

    logger.info("Downloading model weights from %s to %s...", storage_uri, model_path)

    if not storage_uri.startswith("gs://"):
        logger.error("AIP_STORAGE_URI must be a valid GCS path starting with gs://")
        sys.exit(1)

    path_parts = storage_uri[5:].split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""

    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage package is not installed. Cannot download weights.")
        sys.exit(1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    pt_blob = None
    for blob in blobs:
        if blob.name.endswith(".pt") or blob.name.endswith(".ckpt") or blob.name.endswith(".bin"):
            pt_blob = blob
            break

    if not pt_blob:
        blob = bucket.get_blob(prefix)
        if blob and (prefix.endswith(".pt") or prefix.endswith(".ckpt") or prefix.endswith(".bin")):
            pt_blob = blob

    if not pt_blob:
        logger.error("No model weight checkpoint file found at GCS path %s", storage_uri)
        sys.exit(1)

    logger.info("Found checkpoint file: gs://%s/%s", bucket_name, pt_blob.name)
    pt_blob.download_to_filename(model_path)
    logger.info("Model weights download complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_model_weights()
