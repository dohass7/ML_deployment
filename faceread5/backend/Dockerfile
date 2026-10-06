FROM python:3.11-slim

# ── Variables d'environnement ─────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Dépendances système ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Workdir ───────────────────────────────────────────────────────
WORKDIR /app

# ── Torch CPU-only (AVANT requirements.txt) ───────────────────────
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 torchvision==0.23.0

# ── Reste des dépendances ─────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Télécharger le modèle ViT pendant le BUILD ────────────────────
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='michaelgathara/vit-face-raf-db', local_dir='./vit-face-raf-db')"

# ── Code source + frontend ────────────────────────────────────────
COPY backend/app.py backend/model.py backend/database.py backend/face_detection_yunet_2023mar.onnx ./

# ── Port
EXPOSE 8000

# ── Démarrage 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]