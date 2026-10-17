# ============================================================
# STAGE 1 : BUILDER
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances de BUILD (uniquement dans ce stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# PyTorch CPU + torchvision CPU (versions alignées et disponibles)
# --index-url force la wheel CPU pour éviter l'erreur "torchvision::nms"
RUN pip install --user --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 \
    torchvision==0.20.1

# Reste des dépendances
RUN pip install --user --no-cache-dir -r requirements.txt


# ============================================================
# STAGE 2 : RUNTIME
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dépendances RUNTIME uniquement (pas de compilateurs !)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libheif1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les packages Python installés dans le builder
COPY --from=builder /root/.local /root/.local

# Copier le code applicatif
COPY backend/app.py .
COPY backend/model.py .
COPY backend/database.py .
COPY backend/schemas.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ⚠️ 1 seul worker : LLM en RAM, pas de duplication
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]