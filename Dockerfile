FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with Google Drive support
RUN pip install "dvc[gdrive]"

# Configure DVC remote to use service account auth
# (the JSON file path matches what start.sh writes at runtime)
RUN dvc remote modify myremote gdrive_use_service_account true && \
    dvc remote modify myremote gdrive_service_account_json_file_path /tmp/gdrive_sa.json

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 10000

# start.sh writes credentials → dvc pull → gunicorn (all at runtime)
CMD ["./start.sh"]