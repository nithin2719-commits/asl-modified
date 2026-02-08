# Lightweight production image for SignFlow (FastAPI + WebRTC + TensorFlow)
FROM python:3.11-slim

# System deps for opencv/mediapipe/tensorflow runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code, templates, static, model, etc.
COPY . .

CMD ["uvicorn", "camera_feed:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
