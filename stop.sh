#!/bin/bash
set -e

APP_NAME="synthline"

CONTAINER_ID=$(docker ps -aq -f "name=^/${APP_NAME}$")

if [ -n "$CONTAINER_ID" ]; then
    echo "Stopping $APP_NAME..."
    docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_ID"
    echo "Stopped."
else
    echo "$APP_NAME does not exist."
fi