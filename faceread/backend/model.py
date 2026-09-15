import os

# Désactive le GPU (tu es sur CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

# ── Chargement du modèle d'émotion ──
model = keras.models.load_model("best_model.h5")
labels = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Chargement du détecteur YuNet ──
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.7,   # seuil de confiance
    0.3,   # NMS
    5000   # top_k
)


def preprocess_image(pil_image):
    """Reçoit un objet PIL.Image déjà ouvert."""
    image = pil_image.convert("L")
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    face_array = np.array(img).reshape(-1, 128, 128, 1)
    face = tf.keras.utils.normalize(face_array, axis=1)
    face_tf = tf.cast(face, tf.float32)
    return face_tf


def check_status():
    if model is None:
        return "model not loaded"
    return f"ready ({len(model.layers)} layers)"


def predict_emotion(face_array):
    preds = model.predict(face_array, verbose=0)
    emotion_class = preds.argmax()
    return labels[emotion_class]


def detect_faces_and_predict(image_bgr, margin=0.1):
    """
    Détecte les visages dans une image et prédit l'émotion de chacun.

    Args:
        image_bgr : image OpenCV (BGR) — np.array
        margin    : marge proportionnelle autour du visage (0.1 = 10 %)

    Returns:
        results : liste de dicts [{'box': (x, y, w, h), 'emotion': str}, ...]
    """
    H, W = image_bgr.shape[:2]

    # ── Resize pour la détection ──
    img_small = cv2.resize(image_bgr, (320, 320))
    h_small, w_small = img_small.shape[:2]

    # ── Détection ──
    face_detector.setInputSize((w_small, h_small))
    _, faces = face_detector.detect(img_small)

    results = []
    if faces is None:
        return results

    # ── Facteurs d'échelle ──
    scale_x = W / w_small
    scale_y = H / h_small

    for face in faces:
        x, y, w_box, h_box = face[:4]

        # Marge proportionnelle
        margin_x = w_box * margin / 2
        margin_y = h_box * margin / 2

        x_new = max(0, x - margin_x)
        y_new = max(0, y - margin_y)
        w_new = min(w_small - x_new, w_box + margin_x * 2)
        h_new = min(h_small - y_new, h_box + margin_y * 2)

        # Remise à l'échelle
        x_orig = int(x_new * scale_x)
        y_orig = int(y_new * scale_y)
        w_orig = int(w_new * scale_x)
        h_orig = int(h_new * scale_y)

        # Crop
        roi = image_bgr[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
        if roi.size == 0:
            continue

        # Prétraitement
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        face_input = preprocess_image(roi_pil)

        # Prédiction
        emotion = predict_emotion(face_input)

        results.append({
            "box": (x_orig, y_orig, w_orig, h_orig),
            "emotion": emotion,
        })

    return results