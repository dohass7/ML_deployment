import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

# Charger le modèle .h5
model = keras.models.load_model("best_model.h5")

# Labels des émotions (à adapter selon ton entraînement)
labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
def check_status():
    """Retourne l'état du modèle local."""
    if model is None:
        return "model not loaded"
    return f"ready ({len(model.layers)} layers, {model.count_params()} params)"

def predict_emotion(face_array):
    # face_array doit être prétraité (taille, normalisation)
    preds = model.predict(face_array)
    emotion_class = preds.argmax()
    return labels[emotion_class]