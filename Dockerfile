FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries (torch, sentence-transformers)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python to unbuffered mode to see logs immediately
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./app /app/app
COPY main.py .

# Run the app - using shell form to allow $PORT expansion
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
