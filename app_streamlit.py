import streamlit as st 
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2

st.title("Face emotion classification")

with st.sidebar:
    st.header('Data requirements')
    st.caption('To inference the model you nead to upload the image')

    with st.expander('Data format'):
        st.markdown(' - png')
        st.markdown(' - jpg')
        st.markdown(' - tiff')
               
    st.divider() 
    st.caption("<p style = 'text-align:center'>Developed by me</p>", unsafe_allow_html = True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

def preprocess_image(uploaded_file):
    # Ouvrir l'image avec PIL
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)

    # Redimensionner à la taille attendue par ton modèle (ex: 256x256)
    img = cv2.resize(img, (32, 32))

    # Normaliser
    img = img / 255.0

    # Ajouter dimension batch
    face_array = np.array(img).reshape(-1,32,32,1)
    face = tf.keras.utils.normalize(face_array, axis=1)
    face_tf = tf.cast(face,tf.float32)
    return face_tf
# Charger le modèle .h5
model = keras.models.load_model("/home/ddb/emotion-api/best_model.h5")

# Labels des émotions (à adapter selon ton entraînement)
labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

def predict_emotion(face_array):
    # face_array doit être prétraité (taille, normalisation)
    preds = model.predict(face_array)
    emotion_class = preds.argmax()
    return labels[emotion_class]

st.button("Let's get started", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Afficher l'image uploadée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_column_width=True)

        # Prétraitement
        face_array = preprocess_image(uploaded_file)

        # Prédiction
        
        emotion = predict_emotion(face_array)

        st.header("Émotion prédite")
        st.write(emotion)