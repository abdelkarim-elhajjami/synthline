#!/bin/bash

set -e

if docker-compose up -d --build; then
  echo "Deployment completed successfully!"
  echo "Web application is running at: http://localhost:3000"
  echo "API is running at: http://localhost:8000"
else
  EXIT_CODE=$?
  echo "Deployment failed with exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi 