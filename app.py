import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


st.title("Classificador dos Chapéus de Palha - One Piece")

@st.cache_resource  
def load_my_model():
    return load_model('one_piece_v3.keras')

model = load_my_model()

characters = ['Luffy', 'Zoro', 'Nami', 'Usopp', 'Sanji', 'Chopper', 'Robin', 'Franky', 'Brook', 'Jinbe']


def predict_character(img):
    img = img.resize((128, 128))  # redimensionar a imagem para o tamanho esperado pelo modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalizar a imagem
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    return characters[predicted_class[0]], prediction[0]

# upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # carregar a imagem
    img = Image.open(uploaded_file)
    
    # exibir a imagem
    st.image(img, caption='Imagem carregada', use_column_width=True)
    
    # fazer a previsão
    st.write("Classificando...")
    predicted_character, confidence_scores = predict_character(img)
    
    # exibir o resultado
    st.success(f"O personagem na imagem é: **{predicted_character}**")
    