#!/bin/bash
# scripts/start-separate.sh

set -e

echo "Creating Docker network..."
docker network create jaybird-network 2>/dev/null || echo "Network already exists"

echo "Building images..."
docker build -f Dockerfile.inference -t jaybird-inference:latest .
docker build -f Dockerfile.go -t jaybird-api:latest .

echo "Starting inference service..."
docker stop inference-service 2>/dev/null || true
docker rm inference-service 2>/dev/null || true

docker run -d \
  --name inference-service \
  --network jaybird-network \
  -p 8000:8000 \
  -v $(pwd)/model_registry:/app/model_registry \
  jaybird-inference:latest

echo "Waiting for inference service to be ready..."
sleep 5

echo "Starting API service..."
docker stop api-service 2>/dev/null || true
docker rm api-service 2>/dev/null || true

# Load environment variables from .env file
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

docker run -d \
  --name api-service \
  --network jaybird-network \
  -p 8080:8080 \
  -e PORT=8080 \
  -e CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME} \
  -e CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY} \
  -e CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET} \
  -e CORS_ALLOWED_URL=${CORS_ALLOWED_URL:-http://localhost:3000} \
  -e PYTHON_INFERENCE_URL=http://inference-service:8000 \
  -e TEMP_DIR=/tmp \
  jaybird-api:latest

echo ""
echo "✅ Both services are running!"
echo "  - API: http://localhost:8080"
echo "  - Inference: http://localhost:8000"
echo ""
echo "View logs:"
echo "  docker logs -f api-service"
echo "  docker logs -f inference-service"