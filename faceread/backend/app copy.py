import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io
from PIL import Image
from model import predict_emotion, check_status, preprocess_image

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
    return {
        "status": "healthy",
        "message": "Backend service is running",
        "model_status": status,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    face_tf = preprocess_image(image)

    emotion = predict_emotion(face_tf)
    return {"emotion": emotion}