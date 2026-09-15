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
    predict_emotion,
    check_status,
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

    image = Image.open(
        io.BytesIO(contents)
    )

    results = predict_emotion(image)

    return {
        "faces": results
    }