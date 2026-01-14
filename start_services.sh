#!/bin/bash

# Start Python inference service in background
cd ml
source venv/bin/activate

# Set PYTHONPATH so Python can find the ml module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting Python inference service on port 8000..."
uvicorn inference.predictor:app --host 0.0.0.0 --port 8000 --reload &
PYTHON_PID=$!

# Wait a bit for Python to start
sleep 3

# Start Go backend
cd ..
echo "Starting Go backend on port 8080..."
go run cmd/api/main.go &
GO_PID=$!

echo "Both services started!"
echo "Python PID: $PYTHON_PID"
echo "Go PID: $GO_PID"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap "kill $PYTHON_PID $GO_PID" EXIT
wait