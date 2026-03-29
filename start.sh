#!/bin/bash
set -e

echo "============================================"
echo " SmartWaste AI – Container Startup"
echo "============================================"

# Check env var
if [ -z "$GDRIVE_CREDENTIALS_DATA" ]; then
  echo "ERROR: GDRIVE_CREDENTIALS_DATA env var is not set."
  exit 1
fi

# Write credentials
echo "$GDRIVE_CREDENTIALS_DATA" > /tmp/gdrive_sa.json
echo "[1/4] GDrive credentials written."

# 🔥 CONFIGURE DVC AUTH (MISSING FIX)
echo "[2/4] Configuring DVC service account..."
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path /tmp/gdrive_sa.json

# Pull model
echo "[3/4] Running dvc pull..."
dvc pull
echo "      Model downloaded successfully."

# Start app
echo "[4/4] Starting gunicorn..."
exec gunicorn app.app:app \
  --bind 0.0.0.0:10000 \
  --workers 1 \
  --timeout 120 \
  --log-level info