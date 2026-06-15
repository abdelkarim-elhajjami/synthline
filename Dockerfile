# -----------------------------------------------------------------------------
# Synthline Setup for Hugging Face Spaces
# -----------------------------------------------------------------------------

# Stage 1: Build the Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /web

# Deployment: 'local' for local dev (shows Ollama), 'hf' for HF Spaces (shows HF models)
ARG NEXT_PUBLIC_DEPLOYMENT='hf'
ENV NEXT_PUBLIC_DEPLOYMENT=${NEXT_PUBLIC_DEPLOYMENT}

# Copy package files
COPY web/package.json web/package-lock.json ./
# Install dependencies
RUN npm ci
# Copy source code
COPY web/ .
# Build static site (output goes to /web/out)
RUN npm run build

# Stage 2: Setup Backend and Final Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY requirements.txt .

# Install CPU-only PyTorch first so CPU deployments do not pull CUDA runtimes.
ARG TORCH_VERSION=2.8.0
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /home/appuser/.cache/huggingface

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /home/appuser

# Copy Backend Code
COPY synthline/ synthline/
COPY server/ server/

# Copy Frontend Build from Stage 1
COPY --from=frontend-builder /web/out /app/server/static

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/server

# Expose the port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Run the application
WORKDIR /app/server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
