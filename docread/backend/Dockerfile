# ============================================================
# STAGE 1 : BUILDER
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# PyTorch CPU + torchvision : versions compatibles
RUN pip install --user --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 \
    torchvision==0.21.0

# Vérification immédiate
RUN python -c "import torch, torchvision; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__)"

# Autres dépendances
RUN pip install --user --no-cache-dir -r requirements.txt

# Vérification après installation des autres dépendances
RUN python -c "import torch, torchvision, transformers; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Transformers:', transformers.__version__)" \
    && pip check


# ============================================================
# STAGE 2 : RUNTIME
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libheif1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

# Vérification dans l'image finale
RUN python -c "import torch, torchvision, transformers; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Transformers:', transformers.__version__)"

COPY backend/app.py .
COPY backend/model.py .
COPY backend/database.py .
COPY backend/schemas.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]