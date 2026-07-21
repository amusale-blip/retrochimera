# RetroChimera Frontier Model - Cloud Build Deployment Commands

This guide outlines the step-by-step terminal commands required to authenticate, set up, and deploy the RetroChimera (`retrochimera`) model on GCP as a Vertex AI custom model container using Google Cloud Build, leveraging existing shared infrastructure (`moltrans-containers` Artifact Registry & `x-woodward-moltrans-model-store` GCS Bucket).

---

### Step 1: Navigate to the Model Directory
Ensure your terminal is in the directory containing the model source code, `Dockerfile`, and `.gcloudignore` files:
```bash
cd ~/Downloads/retrochimera
```

### Step 2: Install Google Cloud CLI (if missing)
If the `gcloud` CLI tool is not installed on your local machine, run the following to install it:
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
gcloud config set project x-woodward
```

### Step 5: Enable Required GCP APIs
Ensure Cloud Build, Artifact Registry, AI Platform, and Storage APIs are active:
```bash
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com storage.googleapis.com
```

### Step 6: Build and Register the Image in Artifact Registry
Submit local files to Google Cloud Build. This remote build packages the source code, compiles the Dockerfile, and pushes the image into the existing `moltrans-containers` Artifact Registry repository under tag `retrochimera-service:latest`:
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/x-woodward/moltrans-containers/retrochimera-service:latest .
```

---

## Phase 4: Vertex AI Registration & Model Deployment

### Step 7: Upload Model Weights (`test_model.pt`) to Google Cloud Storage
Upload the test model checkpoint file directly to the shared GCS bucket under the dedicated subfolder prefix (`retrochimera/`):
```bash
# Copy test_model.pt checkpoint straight to GCS under retrochimera prefix
gcloud storage cp ~/Downloads/x-woodward-investigations/onmt_MolTrans/src/moltrans_onmt/tests/test_model.pt gs://x-woodward-moltrans-model-store/retrochimera/checkpoint.pt
```

### Step 8: Register Model inside Vertex AI Model Registry
Register the pushed custom Docker container and bind its GCS weight directory explicitly (`--artifact-uri`), along with exact port & route specs required by Vertex AI:
```bash
gcloud ai models upload \
    --region=us-central1 \
    --display-name="retrochimera-service-model" \
    --container-image-uri="us-central1-docker.pkg.dev/x-woodward/moltrans-containers/retrochimera-service:latest" \
    --artifact-uri="gs://x-woodward-moltrans-model-store/retrochimera/" \
    --container-predict-route="/predict" \
    --container-health-route="/health" \
    --container-ports=8080
```
*(Note down the returned **Model ID** or verify via `gcloud ai models list --region=us-central1`)*

### Step 9: Create a Vertex AI Prediction Endpoint
Provision a dedicated target endpoint inside Vertex AI for RetroChimera:
```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="retrochimera-endpoint"
```
*(Note down the returned **Endpoint ID** or view it with `gcloud ai endpoints list --region=us-central1`)*

### Step 10: Deploy Model to the Endpoint with Auto-Scaling & GPU Support
Deploy your registered model directly onto standard `n1-standard-4` hardware running an **NVIDIA_TESLA_T4** GPU across scale-to-zero autoscaling boundaries (`--min-replica-count=0`):
```bash
# Replace [RETROCHIMERA_ENDPOINT_ID] and [RETROCHIMERA_MODEL_ID] below with numeric IDs from Steps 8 & 9
gcloud ai endpoints deploy-model 2012551031283515392 \
    --region=us-central1 \
    --model=7588794620293677056 \
    --display-name="retrochimera-deployment" \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --min-replica-count=0 \
    --max-replica-count=2 \
    --traffic-split=0=100
```

---

## Phase 5: Post-Deployment Setup & Testing Guide

### Test Option A: Verification via `gcloud` CLI (Sanity Test)
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
Run prediction using your target numeric Endpoint ID:
```bash
gcloud ai endpoints predict 2012551031283515392 \
    --region=us-central1 \
    --json-request=test_instances.json
```

---

## Phase 6: Resource Management & Scale-to-Zero

Because `--min-replica-count=0` is active:
- **Automatic Scale-to-Zero**: If zero prediction requests reach the endpoint for ~15 to 20 minutes, Vertex AI deallocates VM instances, bringing compute charges to **$0.00/hour**.
- **Manual Immediate Shutdown**: To undeploy manually:
```bash
# Get active deployed model ID
gcloud ai endpoints describe 2012551031283515392 --region=us-central1 --format="value(deployedModels.id)"

# Undeploy model
gcloud ai endpoints undeploy-model 2012551031283515392 --region=us-central1 --deployed-model-id=1730706618665926656
```
