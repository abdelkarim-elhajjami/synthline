#!/bin/bash
set -e

APP_NAME="synthline"

if [ "$(docker ps -q -f name=$APP_NAME)" ]; then
    echo "Stopping $APP_NAME..."
    docker stop $APP_NAME
    echo "Stopped."
else
    echo "$APP_NAME is not running."
fi
