import streamlit as st 
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

st.title("Face emotion classification")

with st.sidebar:
    st.header('Data requirements')
    st.caption('To inference the model you nead to upload the image')

    with st.expander('Data format'):
        st.markdown(' - png')
        st.markdown(' - jpg')
        st.markdown(' - tiff')
               
    st.divider() 
    st.caption("<p style = 'text-align:center'>Developed by doh</p>", unsafe_allow_html = True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

def preprocess_image(uploaded_file):
    
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)

    img = cv2.resize(img, (32, 32))

    img = img / 255.0
    
    face_array = np.array(img).reshape(-1,32,32,1)
    face = tf.keras.utils.normalize(face_array, axis=1)
    face_tf = tf.cast(face,tf.float32)
    return face_tf

def predict_emotion(face_array):
    
    preds = model.predict(face_array)
    emotion_class = preds.argmax()
    return labels[emotion_class]

BASE_DIR = Path(__file__).resolve().parent

ROOT_DIR = BASE_DIR.parent

MODEL_PATH = ROOT_DIR / "best_model.h5"

if not MODEL_PATH.exists():
    import streamlit as st
    st.error(f"Le fichier de modèle est introuvable : {MODEL_PATH}")
    st.write("Contenu du répertoire racine du dépôt :", list(ROOT_DIR.iterdir()))
    st.stop()

model = keras.models.load_model(MODEL_PATH)
#model = keras.models.load_model("best_model.h5")

labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']


st.button("Let's get started", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_column_width=True)

        face_array = preprocess_image(uploaded_file)
        
        emotion = predict_emotion(face_array)

        st.header("Émotion prédite")
        st.write(emotion)