FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Use a clean InsightFace cache
ENV INSIGHTFACE_HOME=/root/.insightface

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
