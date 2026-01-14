#!/bin/bash
# scripts/stop-separate.sh

echo "Stopping services..."
docker stop api-service inference-service 2>/dev/null || true
docker rm api-service inference-service 2>/dev/null || true

echo "✅ Services stopped"