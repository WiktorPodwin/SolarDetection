#!/bin/bash

set -e

echo "Starting the app"
exec uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1 --log-level debug --reload