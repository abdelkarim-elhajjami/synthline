#!/bin/bash
set -e

# 1. Build (for local development)
echo "Building..."
docker build --build-arg NEXT_PUBLIC_DEPLOYMENT=local -t synthline:latest .

# 2. Cleanup
docker rm -f synthline 2>/dev/null || true

# 3. Run
echo "Running..."
docker run -d \
  -p 3000:7860 \
  --env-file .env \
  -v "hf_cache:/home/appuser/.cache/huggingface" \
  --name synthline \
  --restart unless-stopped \
  synthline:latest

echo "Synthline is running at: http://localhost:3000" 