#!/bin/bash
set -e

# 1. Build
echo "Building..."
docker build -t synthline:latest .

# 2. Cleanup
docker rm -f synthline 2>/dev/null || true

# 3. Run
echo "Running..."
docker run -d \
  -p 3000:7860 \
  --env-file engine/.env \
  -v "hf_cache:/home/appuser/.cache/huggingface" \
  --name synthline \
  --restart unless-stopped \
  synthline:latest

echo "Synthline is running at: http://localhost:3000" 