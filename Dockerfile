FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY app.py .
COPY config.yaml .
COPY requirements-app.txt .

# Install Python deps (lightweight for inference only)
RUN pip install --no-cache-dir -r requirements-app.txt

# Expose port
EXPOSE 7860

# Run Gradio app
CMD ["python", "app.py"]
