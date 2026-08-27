FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV INSIGHTFACE_HOME=/root/.insightface

# Download and extract InsightFace model
RUN mkdir -p /root/.insightface/models && \
    wget -q \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    -O /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models && \
    rm /tmp/buffalo_l.zip

# Show what was actually extracted
RUN find /root/.insightface/models -maxdepth 3 -type f -name "*.onnx" -print

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
