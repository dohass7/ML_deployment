import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

model = keras.models.load_model("best_model.h5")

labels = [
    "anger",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.7,
    0.3,
    5000
)

def check_status():
    if model is None:
        return "model not loaded"

    return f"ready ({len(model.layers)} layers)"

def predict_emotion(pil_image):

    img = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    H, W = img.shape[:2]

    img_small = cv2.resize(img, (320, 320))

    h_small, w_small = img_small.shape[:2]

    face_detector.setInputSize((w_small, h_small))

    _, faces = face_detector.detect(img_small)

    if faces is None or len(faces) == 0:
        return "no_face"
    print("Nombre de visages détectés :", 0 if faces is None else len(faces))
    scale_x = W / w_small
    scale_y = H / h_small

    #face = faces[0]

    results = []

    for i, face in enumerate(faces, start=1):

        x, y, w_box, h_box = face[:4]

        margin = 0.1

        margin_x = w_box * margin / 2
        margin_y = h_box * margin / 2

        x_new = max(0, x - margin_x)
        y_new = max(0, y - margin_y)

        w_new = min(w_small - x_new, w_box + margin_x * 2)
        h_new = min(h_small - y_new, h_box + margin_y * 2)

        x_orig = int(x_new * scale_x)
        y_orig = int(y_new * scale_y)

        w_orig = int(w_new * scale_x)
        h_orig = int(h_new * scale_y)

        roi = img[
            y_orig:y_orig+h_orig,
            x_orig:x_orig+w_orig
        ]

        if roi.size == 0:
            return "invalid_face"

        roi_gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        face_input = cv2.resize(
            roi_gray,
            (128, 128)
        )

        face_input = (
            face_input.astype("float32")
            / 255.0
        )

        face_input = np.expand_dims(
            face_input,
            axis=-1
        )

        face_input = np.expand_dims(
            face_input,
            axis=0
        )

        pred = model.predict(face_input, verbose=0)

        emotion = labels[np.argmax(pred)]

        results.append({
            "face": i,
            "emotion": emotion
        })
