import os

# Désactive le GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import base64
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best_model.h5"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

IMAGE_SIZE = 128

labels = [
    "anger",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# CHARGEMENT DU MODÈLE EMOTION
# ============================================================

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modèle émotion chargé : {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"ERREUR chargement modèle émotion : {e}")


# ============================================================
# CHARGEMENT DE YUNET
# ============================================================

try:
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    print(f"YuNet chargé : {YUNET_MODEL_PATH}")

except Exception as e:
    face_detector = None
    print(f"ERREUR chargement YuNet : {e}")


# ============================================================
# STATUS
# ============================================================

def check_status():

    if model is None:
        return "model not loaded"

    if face_detector is None:
        return "YuNet not loaded"

    return (
        f"ready "
        f"(emotion model: {len(model.layers)} layers, "
        f"YuNet: active)"
    )


# ============================================================
# CONVERSION PIL -> OPENCV
# ============================================================

def pil_to_cv2(pil_image):
    """
    Convertit une image PIL en image OpenCV BGR.
    """

    image_rgb = pil_image.convert("RGB")

    image_np = np.array(image_rgb)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    return image_bgr


# ============================================================
# PREPROCESSING D'UN VISAGE
# ============================================================

def preprocess_face(face):

    # Passage en niveaux de gris
    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Resize 128x128
    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalisation
    gray = gray.astype(np.float32) / 255.0

    # Shape :
    # (128,128)
    # ->
    # (128,128,1)
    # ->
    # (1,128,128,1)

    face_input = np.expand_dims(
        gray,
        axis=-1
    )

    face_input = np.expand_dims(
        face_input,
        axis=0
    )

    face_input = tf.cast(
        face_input,
        tf.float32
    )

    return face_input


# ============================================================
# COULEURS DES RECTANGLES
# ============================================================

def get_color(emotion):

    colors = {
        "anger": (0, 0, 255),
        "fear": (128, 0, 128),
        "happy": (0, 255, 0),
        "neutral": (255, 255, 255),
        "sad": (255, 0, 0),
        "surprise": (0, 165, 255)
    }

    return colors.get(
        emotion.lower(),
        (0, 255, 0)
    )


# ============================================================
# DESSIN DU LABEL
# ============================================================

def draw_label(
    image,
    x,
    y,
    text,
    color
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    # Position du fond du label
    label_x1 = x
    label_y1 = max(
        0,
        y - text_height - baseline - 10
    )

    label_x2 = x + text_width + 12
    label_y2 = y

    # Fond
    cv2.rectangle(
        image,
        (
            label_x1,
            label_y1
        ),
        (
            label_x2,
            label_y2
        ),
        color,
        -1
    )

    # Texte noir pour bonne lisibilité
    cv2.putText(
        image,
        text,
        (
            x + 6,
            y - 6
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92
        ]
    )

    if not success:
        raise RuntimeError(
            "Impossible d'encoder l'image annotée"
        )

    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return image_base64


# ============================================================
# PRÉDICTION
# ============================================================

def predict_emotion(pil_image):

    if model is None:
        raise RuntimeError(
            "Le modèle émotion n'est pas chargé."
        )

    if face_detector is None:
        raise RuntimeError(
            "Le détecteur YuNet n'est pas chargé."
        )

    # --------------------------------------------------------
    # Image originale
    # --------------------------------------------------------

    image = pil_to_cv2(pil_image)

    original_height, original_width = image.shape[:2]

    # --------------------------------------------------------
    # Image pour YuNet
    # --------------------------------------------------------

    detection_width = 320
    detection_height = 320

    detection_image = cv2.resize(
        image,
        (
            detection_width,
            detection_height
        )
    )

    # --------------------------------------------------------
    # Configuration YuNet
    # --------------------------------------------------------

    face_detector.setInputSize(
        (
            detection_width,
            detection_height
        )
    )

    # --------------------------------------------------------
    # Détection
    # --------------------------------------------------------

    _, faces = face_detector.detect(
        detection_image
    )

    # --------------------------------------------------------
    # Aucun visage
    # --------------------------------------------------------

    if faces is None:

        print("Nombre de visages détectés : 0")

        return {
            "faces": [],
            "image": image_to_base64(image)
        }

    print(
        f"Nombre de visages détectés : {len(faces)}"
    )

    results = []

    # ========================================================
    # TRAITEMENT DE TOUS LES VISAGES
    # ========================================================

    for index, face_data in enumerate(
        faces,
        start=1
    ):

        # YuNet renvoie :
        #
        # x
        # y
        # width
        # height
        #
        # + landmarks

        x = float(face_data[0])
        y = float(face_data[1])
        w = float(face_data[2])
        h = float(face_data[3])

        # ----------------------------------------------------
        # Conversion coordonnées 320x320 -> image originale
        # ----------------------------------------------------

        scale_x = original_width / detection_width
        scale_y = original_height / detection_height

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # ----------------------------------------------------
        # Sécurité
        # ----------------------------------------------------

        x = max(0, x)
        y = max(0, y)

        x2 = min(
            original_width,
            x + w
        )

        y2 = min(
            original_height,
            y + h
        )

        # Vérification
        if x2 <= x or y2 <= y:
            continue

        # ----------------------------------------------------
        # Crop du visage
        # ----------------------------------------------------

        face_crop = image[
            y:y2,
            x:x2
        ]

        if face_crop.size == 0:
            continue

        # ----------------------------------------------------
        # Préprocessing
        # ----------------------------------------------------

        face_input = preprocess_face(
            face_crop
        )

        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        prediction = model.predict(
            face_input,
            verbose=0
        )

        emotion_index = int(
            np.argmax(prediction)
        )

        emotion = labels[
            emotion_index
        ]

        confidence = float(
            np.max(prediction)
        )

        confidence_percent = (
            confidence * 100
        )

        print(
            f"Visage {index} : "
            f"{emotion} "
            f"({confidence_percent:.2f}%)"
        )

        # ----------------------------------------------------
        # Résultat JSON
        # ----------------------------------------------------

        results.append({
            "face": index,
            "emotion": emotion,
            "confidence": confidence
        })

        # ----------------------------------------------------
        # Couleur
        # ----------------------------------------------------

        color = get_color(
            emotion
        )

        # ----------------------------------------------------
        # Rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x, y),
            (x2, y2),
            color,
            3
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"Visage {index} | "
            f"{emotion.capitalize()} | "
            f"{confidence_percent:.1f}%"
        )

        draw_label(
            image,
            x,
            y,
            label,
            color
        )

    # ========================================================
    # IMAGE FINALE
    # ========================================================

    annotated_image = image_to_base64(
        image
    )

    return {
        "faces": results,
        "image": annotated_image
    }