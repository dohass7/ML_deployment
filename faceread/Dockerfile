FROM python:3.11-slim

# ── Env ──────────────────────────────────────────────────────────
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Dépendances système ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Workdir ───────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances Python ────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source + frontend ────────────────────────────────────────
COPY backend/app.py backend/model.py backend/best_model.h5 backend/face_detection_yunet_2023mar.onnx ./
COPY frontend/ ./frontend/

# ── Port ──────────────────────────────────────────────────────────
EXPOSE 8000

# ── Démarrage ─────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
