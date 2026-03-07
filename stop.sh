#!/bin/bash
set -e

APP_NAME="synthline"
IMAGE_NAME="synthline:latest"

CONTAINER_ID=$(docker ps -aq -f "name=^/${APP_NAME}$")
if [ -n "$CONTAINER_ID" ]; then
    docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
fi

if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    docker rmi -f "$IMAGE_NAME" >/dev/null 2>&1 || true
fi
docker image prune -f >/dev/null 2>&1 || true
docker volume prune -f >/dev/null 2>&1 || true
docker builder prune --all -f >/dev/null 2>&1 || true

echo "Synthline stopped."