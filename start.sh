#!/bin/bash
set -e

echo "============================================"
echo " SmartWaste AI – Container Startup"
echo "============================================"

# Write the GDrive service account credentials from env var to a temp file
if [ -z "$GDRIVE_CREDENTIALS_DATA" ]; then
  echo "ERROR: GDRIVE_CREDENTIALS_DATA env var is not set."
  echo "Set it in Render → Environment → Add Environment Variable."
  exit 1
fi

echo "$GDRIVE_CREDENTIALS_DATA" > /tmp/gdrive_sa.json
echo "[1/3] GDrive credentials written."

# Pull model from DVC remote
echo "[2/3] Running dvc pull..."
dvc pull
echo "      Model downloaded successfully."

# Start the Flask app via gunicorn
echo "[3/3] Starting gunicorn..."
exec gunicorn app.app:app \
  --bind 0.0.0.0:10000 \
  --workers 1 \
  --timeout 120 \
  --log-level info
