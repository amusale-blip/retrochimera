# ==============================================================================
# RetroChimera Frontier Model Container Build Specification
# Base Image: PyTorch 2.1 CUDA 11.8 Runtime with CuDNN 8
# Purpose: Containerizes Microsoft RetroChimera ensemble retrosynthesis model for
#          GCP Vertex AI scale-to-zero microservice deployments.
# ==============================================================================

# Use PyTorch with CUDA support as base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Copy the model folder contents
COPY . /app

# Set pretend version for setuptools_scm when building in container without .git folder
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    google-cloud-storage \
    rdkit \
    pydantic

# Install local package in editable mode using pretend version
RUN pip install --no-build-isolation -e .

# Expose default Vertex AI HTTP Port
EXPOSE 8080

# Environment variables defaults for Vertex AI container runtime
ENV AIP_HTTP_PORT=8080
ENV RETROCHIMERA_MODEL_PATH=checkpoint.pt

# Entrypoint script: downloads GCS weights first, then runs FastAPI server
ENTRYPOINT ["sh", "-c", "python3 download_weights.py && uvicorn app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
