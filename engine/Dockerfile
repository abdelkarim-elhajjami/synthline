# -----------------------------------------------------------------------------
# Synthline Engine Dockerfile
# -----------------------------------------------------------------------------
# Build stage for compiling dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /engine

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/engine

# Create non-root user and necessary directories
RUN useradd -m -s /bin/bash appuser \
    && mkdir -p /engine/logs /engine/output \
    && chown -R appuser:appuser /engine

# Copy only the necessary files
COPY *.py .
COPY .env* ./

# Set permissions for output and log directories
RUN chown -R appuser:appuser /engine/logs /engine/output \
    && chmod 755 /engine/logs /engine/output

# Declare volumes for persistent data
VOLUME ["/engine/logs", "/engine/output"]

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://0.0.0.0:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "routes:app", "--host", "0.0.0.0", "--port", "8000"]