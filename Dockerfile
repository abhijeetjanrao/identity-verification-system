FROM python:3.10-slim

WORKDIR /app

# System libraries required by OpenCV / InsightFace
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . ./

# Create InsightFace model directory
RUN mkdir -p /root/.insightface/models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
