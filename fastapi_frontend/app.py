import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Désactive le GPU

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
from model import predict_emotion, check_status

app = FastAPI(title="Face Emotion API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir le frontend statique
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
    return {
        "status": "healthy",
        "message": "Backend service is running",
        "sagemaker_endpoint": status,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image envoyée
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Image invalide ou non lisible"}

    # Prétraitement : 32x32, normalisation
    face = cv2.resize(img, (32, 32))
    face = face / 255.0
    face_array = np.array(face).reshape(-1, 32, 32, 1)
    face_tf = tf.cast(face_array, tf.float32)

    # Prédiction via SageMaker
    emotion = predict_emotion(face_tf.numpy())
    return {"emotion": emotion}
