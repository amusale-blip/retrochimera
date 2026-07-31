# RetroChimera Frontier Model - Cloud Build & Vertex AI Deployment Guide

This guide outlines the step-by-step terminal commands required to authenticate, set up, and deploy the RetroChimera (`retrochimera`) model on GCP as a Vertex AI custom model container using Google Cloud Build.

---

### Step 1: Navigate to the Model Directory
Ensure your terminal is in the directory containing the model source code, `Dockerfile`, and `.gcloudignore` files:
```bash
cd ~/Downloads/retrochimera
```

### Step 2: Install Google Cloud CLI (if missing)
If the `gcloud` CLI tool is not installed on your local machine, run:
```bash
sudo apt-get update && sudo apt-get install -y google-cloud-cli
```

### Step 3: Log In to Google Cloud
Authenticate your local CLI session with your Google credentials:
```bash
gcloud auth login
```

### Step 4: Configure Target GCP Project
Set the active project property to target your specific Google Cloud Project:
```bash
gcloud config set project ${GCP_PROJECT_ID:-x-woodward}
```

### Step 5: Enable Required GCP APIs
Ensure Cloud Build, Artifact Registry, AI Platform, and Storage APIs are active:
```bash
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com storage.googleapis.com
```

### Step 6: Build and Register the Image in Artifact Registry
Submit local files to Google Cloud Build:
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/${GCP_PROJECT_ID:-x-woodward}/moltrans-containers/retrochimera-service:latest .
```

---

## Phase 4: Vertex AI Registration & Model Deployment

### Step 7: Upload Model Weights to Google Cloud Storage
Upload the model checkpoint file directly to the target GCS bucket under the dedicated subfolder prefix (`retrochimera/`):
```bash
gcloud storage cp path/to/checkpoint.pt gs://${GCP_MODEL_BUCKET:-x-woodward-moltrans-model-store}/retrochimera/checkpoint.pt
```

### Step 8: Register Model inside Vertex AI Model Registry
Register the custom Docker container and bind its GCS weight directory explicitly:
```bash
gcloud ai models upload \
    --region=us-central1 \
    --display-name="retrochimera-service-model" \
    --container-image-uri="us-central1-docker.pkg.dev/${GCP_PROJECT_ID:-x-woodward}/moltrans-containers/retrochimera-service:latest" \
    --artifact-uri="gs://${GCP_MODEL_BUCKET:-x-woodward-moltrans-model-store}/retrochimera/" \
    --container-predict-route="/predict" \
    --container-health-route="/health" \
    --container-ports=8080
```
*(Note down the returned **Model ID**)*

### Step 9: Create a Vertex AI Prediction Endpoint
Provision a target endpoint inside Vertex AI for RetroChimera:
```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="retrochimera-endpoint"
```
*(Note down the returned **Endpoint ID**)*

### Step 10: Deploy Model to the Endpoint with Auto-Scaling & GPU Support
Deploy your registered model directly onto GPU-backed hardware with scale-to-zero autoscaling (`--min-replica-count=0`):
```bash
gcloud ai endpoints deploy-model <RETROCHIMERA_ENDPOINT_ID> \
    --region=us-central1 \
    --model=<RETROCHIMERA_MODEL_ID> \
    --display-name="retrochimera-deployment" \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --min-replica-count=0 \
    --max-replica-count=2 \
    --traffic-split=0=100
```

---

## Phase 5: Testing & Verification

### Test Option: Verification via `gcloud` CLI
Create a JSON input test file `test_instances.json`:
```json
{
  "instances": [
    {
      "smiles": "CC(=O)OCC",
      "n_best": 3
    }
  ]
}
```

Run prediction request using your Endpoint ID:
```bash
gcloud ai endpoints predict <RETROCHIMERA_ENDPOINT_ID> \
    --region=us-central1 \
    --json-request=test_instances.json
```
