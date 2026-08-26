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

# Download InsightFace model during BUILD
RUN mkdir -p /root/.insightface/models && \
    wget -q --show-progress \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    -O /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models/ && \
    rm /tmp/buffalo_l.zip

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
