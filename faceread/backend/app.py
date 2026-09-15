import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io
import base64
from PIL import Image
import numpy as np
import cv2
from model import (
    detect_faces_and_predict,
    check_status,
    predict_emotion,
    preprocess_image,
)

app = FastAPI(title="Face Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health_check():
    try:
        status = check_status()
    except Exception as e:
        status = f"unreachable ({str(e)})"
    return {"status": "healthy", "model_status": status}


# ── Détection + prédiction sur une image complète ──
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Image invalide ou non lisible"}

    # Détection + prédiction sur chaque visage
    results = detect_faces_and_predict(img, margin=0.1)

    # ── Dessiner sur l'image (optionnel, pour l'affichage) ──
    annotated = img.copy()
    for r in results:
        x, y, w, h = r["box"]
        emotion = r["emotion"]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(annotated, emotion, (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # Encoder l'image annotée en base64 (pour le frontend)
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "count": len(results),
        "faces": [
            {"box": list(r["box"]), "emotion": r["emotion"]}
            for r in results
        ],
        "annotated_image": f"data:image/jpeg;base64,{img_base64}",
    }