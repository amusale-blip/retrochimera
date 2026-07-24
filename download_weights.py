"""
Dynamic Google Cloud Storage (GCS) Model Weight Downloader for RetroChimera.
Extracts model weights checkpoint.pt dynamically from AIP_STORAGE_URI at container startup.
Decouples heavy model weights from Docker image layers for fast, scale-to-zero container boots.
"""

import os
import sys

def download_model_weights():
    # AIP_STORAGE_URI is set by Vertex AI (e.g., gs://x-woodward-moltrans-model-store/retrochimera/)
    storage_uri = os.environ.get("AIP_STORAGE_URI")
    print("AIP_STORAGE_URI:", storage_uri)
    model_path = os.environ.get("RETROCHIMERA_MODEL_PATH", "checkpoint.pt")
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)
        os.environ["RETROCHIMERA_MODEL_PATH"] = model_path
    
    if not storage_uri:
        print("Warning: AIP_STORAGE_URI is not set. Bypassing cloud weight download.")
        return

    print(f"Downloading model weights from {storage_uri} to {model_path}...")
    
    if not storage_uri.startswith("gs://"):
        print("Error: AIP_STORAGE_URI must be a valid GCS path (gs://...)")
        sys.exit(1)
        
    path_parts = storage_uri[5:].split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    try:
        from google.cloud import storage
    except ImportError:
        print("Error: google-cloud-storage is not installed. Cannot run downloader.")
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
        print(f"Error: No model weight checkpoint file found at GCS path {storage_uri}")
        sys.exit(1)
        
    print(f"Found checkpoint file: gs://{bucket_name}/{pt_blob.name}")
    pt_blob.download_to_filename(model_path)
    print("Download complete.")

if __name__ == "__main__":
    download_model_weights()
