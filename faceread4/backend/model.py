import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import base64
import numpy as np

from PIL import Image

import torch

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)

from pillow_heif import register_heif_opener

# 1. Enregistre le support HEIC auprès de Pillow (à faire une seule fois)
register_heif_opener()

# ============================================================
# Chargement modèle ViT
# ============================================================

processor = ViTImageProcessor.from_pretrained("./vit-face-raf-db")
model = ViTForImageClassification.from_pretrained("./vit-face-raf-db")

model.eval()
torch.set_num_threads(2) #après les imports pour éviter que PyTorch monopolise tous les CPU du VPS
# ============================================================
# Chargement YuNet
# ============================================================

face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.7,
    0.3,
    5000
)

# ============================================================
# Status
# ============================================================

def check_status():

    return (
        f"ViT ready "
        f"({model.config.num_labels} labels)"
    )

# ============================================================
# Prediction
# ============================================================

def predict_emotion(pil_image):

    img_rgb = np.array(
        pil_image.convert("RGB")
    )

    img = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR
    )

    H, W = img.shape[:2]
    size = 320

    img_small = cv2.resize(
        img,
        (size, size)
    )
    
    face_detector.setInputSize(
        (size, size)
    )

    _, faces = face_detector.detect(
        img_small
    )

    results = []

    if faces is None:

        return {
            "faces": [],
            "image": None
        }

    scale_x = W / size
    scale_y = H / size

    margin = 0.10

    for idx, face in enumerate(faces, start=1):

        x, y, w_box, h_box = face[:4]

        margin_x = w_box * margin / 2
        margin_y = h_box * margin / 2

        x_new = max(0, x - margin_x)
        y_new = max(0, y - margin_y)

        w_new = min(
            320 - x_new,
            w_box + margin_x * 2
        )

        h_new = min(
            320 - y_new,
            h_box + margin_y * 2
        )

        x_orig = int(x_new * scale_x)
        y_orig = int(y_new * scale_y)

        w_orig = int(w_new * scale_x)
        h_orig = int(h_new * scale_y)

        roi = img[
            y_orig:y_orig+h_orig,
            x_orig:x_orig+w_orig
        ]

        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        roi_pil = Image.fromarray(
            roi_rgb
        )

        inputs = processor(
            images=roi_pil,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**inputs)

        logits = outputs.logits

        probs = torch.softmax(
            logits,
            dim=1
        )

        confidence = float(
            probs.max().item()
        )

        predicted_class_idx = int(
            logits.argmax(-1).item()
        )

        emotion = model.config.id2label[
            predicted_class_idx
        ]

        results.append({
            "face": idx,
            "emotion": emotion,
            "confidence": round(
                confidence,
                2
            )
        })

        cv2.rectangle(
            img,
            (x_orig, y_orig),
            (x_orig + w_orig,
             y_orig + h_orig),
            (0, 255, 0),
            4
        )

        cv2.putText(
            img,
            f"{emotion}",
            (x_orig, y_orig - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

    _, buffer = cv2.imencode(
        ".jpg",
        img
    )

    image_b64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return {
        "faces": results,
        "image": image_b64
    }