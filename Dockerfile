# SmartWaste/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# copy project
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# run pipeline first, then API
CMD ["sh", "-c", "python -m src.pipeline && python app/app.py"]