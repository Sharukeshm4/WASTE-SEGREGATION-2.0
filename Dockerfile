FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with Google Drive support
RUN pip install "dvc[gdrive]"

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 10000

# start.sh handles credentials + dvc pull + app start
CMD ["./start.sh"]