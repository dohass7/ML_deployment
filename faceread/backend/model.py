import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

model = keras.models.load_model("best_model.h5")

labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    face_array = np.array(img).reshape(-1, 32, 32, 1)
    face = tf.keras.utils.normalize(face_array, axis=1)
    face_tf = tf.cast(face, tf.float32)
    return face_tf


def check_status():
    if model is None:
        return "model not loaded"
    return f"ready ({len(model.layers)} layers)"


def predict_emotion(face_array):
    preds = model.predict(face_array)
    emotion_class = preds.argmax()
    return labels[emotion_class]