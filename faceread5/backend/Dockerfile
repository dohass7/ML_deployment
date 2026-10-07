FROM python:3.11-slim AS builder

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --user --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 \
    torchvision==0.23.0

RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY backend/app.py .
COPY backend/model.py .
COPY backend/database.py .
COPY backend/face_detection_yunet_2023mar.onnx .

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='michaelgathara/vit-face-raf-db', local_dir='/app/vit-face-raf-db')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]