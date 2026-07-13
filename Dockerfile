FROM python:3.9-slim

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required runtime directories
RUN mkdir -p media logs

# Expose default port (Render/HuggingFace overrides this via PORT env variable)
EXPOSE 8080

CMD ["python", "app.py"]
