# Guarda - Controle de acesso (placas + reconhecimento facial)
# Fase 4: InsightFace (ArcFace) + YOLOv8 + EasyOCR + WebSocket + Prometheus
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# build-essential/g++: insightface Cython extension
# libgomp1: onnxruntime
# ffmpeg: RTSP/RTMP streams via OpenCV CAP_FFMPEG
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Pré-baixar modelos InsightFace (buffalo_sc) para evitar cold-start
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(320, 320)); \
print('InsightFace buffalo_sc pronto.')"

# Pré-baixar modelo YOLOv8 — não-fatal: baixa no primeiro request se falhar aqui
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
path = hf_hub_download(repo_id='keremberke/yolov8n-license-plate-detection', filename='best.pt', token=False); \
from ultralytics import YOLO; \
YOLO(path); \
print('YOLOv8 plate model pronto.')" \
    || echo "YOLO pre-download ignorado — modelo sera baixado no primeiro uso."

# Pré-baixar modelos EasyOCR (English OCR)
RUN python3 -c "\
import easyocr; \
easyocr.Reader(['en'], gpu=False, verbose=False); \
print('EasyOCR pronto.')"

COPY app/ ./app/
COPY main.py .
COPY static/ ./static/

# Para rodar com câmera no host: docker run --device /dev/video0 ...
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
