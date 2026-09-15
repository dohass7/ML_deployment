import os

# Désactive le GPU (tu es sur CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

# ── Chargement du modèle ──
model = keras.models.load_model("best_model.h5")

labels = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Détecteur YuNet (chargé une seule fois) ──
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.5,
    0.3,
    5000
)


# ─────────────────────────────────────
# ÉTAPE 1 : Prétraitement pour le modèle
# ─────────────────────────────────────
def preprocess_image(image_bgr):
    """Reçoit une image BGR (numpy array) et retourne un tenseur prêt pour le modèle."""
    # Convertir en niveaux de gris
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    # Resize à la taille du modèle
    img = cv2.resize(gray, (128, 128))

    # Normalisation
    img = img.astype("float32") / 255.0

    # Dimensions batch + canal
    face_array = np.array(img).reshape(-1, 128, 128, 1)
    face = tf.keras.utils.normalize(face_array, axis=1)
    face_tf = tf.cast(face, tf.float32)

    return face_tf


# ─────────────────────────────────────
# ÉTAPE 2 : Détection des visages
# ─────────────────────────────────────
def detect_faces(image_bgr):
    """Détecte les visages dans une image BGR. Retourne une liste de boxes (x, y, w, h)."""
    H, W = image_bgr.shape[:2]

    # Resize à 128×128 pour la détection
    img_small = cv2.resize(image_bgr, (128, 128))
    h_small, w_small = img_small.shape[:2]

    # Facteurs d'échelle
    scale_x = W / w_small
    scale_y = H / h_small

    # Détection
    face_detector.setInputSize((w_small, h_small))
    _, faces = face_detector.detect(img_small)

    if faces is None:
        return []

    # Marge proportionnelle
    margin = 0.2

    boxes = []
    for face in faces:
        x, y, w_box, h_box = face[:4]

        # Marge proportionnelle (sur l'échelle 128×128)
        margin_x = w_box * margin / 2
        margin_y = h_box * margin / 2

        # Remise à l'échelle sur l'image originale
        x_orig = int((x - margin_x) * scale_x)
        y_orig = int((y - margin_y) * scale_y)
        w_orig = int((w_box + margin_x * 2) * scale_x)
        h_orig = int((h_box + margin_y * 2) * scale_y)

        # Clamp
        x_orig = max(0, x_orig)
        y_orig = max(0, y_orig)
        w_orig = min(w_orig, W - x_orig)
        h_orig = min(h_orig, H - y_orig)

        boxes.append((x_orig, y_orig, w_orig, h_orig))

    return boxes


# ─────────────────────────────────────
# ÉTAPE 3 : Prédiction de l'émotion
# ─────────────────────────────────────
def predict_emotion(face_array):
    """Prédit l'émotion à partir d'un tenseur prétraité."""
    preds = model.predict(face_array, verbose=0)
    emotion_class = preds.argmax()
    return labels[emotion_class]


# ─────────────────────────────────────
# FONCTION PRINCIPALE : tout-en-un
# ─────────────────────────────────────
def predict_faces(image_bgr):
    """Détecte les visages et prédit l'émotion pour chacun.

    Retourne une liste de dicts :
    [{"box": (x, y, w, h), "emotion": "happy"}, ...]
    """
    boxes = detect_faces(image_bgr)

    results = []
    for (x, y, w, h) in boxes:
        roi = image_bgr[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        face_tf = preprocess_image(roi)
        emotion = predict_emotion(face_tf)

        results.append({
            "box": (x, y, w, h),
            "emotion": emotion,
        })

    return results


# ─────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────
def check_status():
    if model is None:
        return "model not loaded"
    return f"ready ({len(model.layers)} layers)"