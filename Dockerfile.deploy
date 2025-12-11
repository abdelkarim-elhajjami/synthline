# -----------------------------------------------------------------------------
# Synthline Setup for Hugging Face Spaces
# -----------------------------------------------------------------------------

# Stage 1: Build the Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /web
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
COPY engine/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/output /home/appuser/.cache/huggingface

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /home/appuser

# Copy Backend Code
COPY engine/ .

# Copy Frontend Build from Stage 1
# We copy it to a 'static' directory which FastAPI will serve
COPY --from=frontend-builder /web/out /app/static

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose the port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Command to run the application
# We listen on 0.0.0.0 and port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
