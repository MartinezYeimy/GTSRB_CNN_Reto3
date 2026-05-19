import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Cargar modelo
model = load_model("cnn_gtsrb.h5")

# Nombres de las clases
nombres_clases = {
    0: "Límite 20 km/h", 1: "Límite 30 km/h", 2: "Límite 50 km/h",
    3: "Límite 60 km/h", 4: "Límite 70 km/h", 5: "Límite 80 km/h",
    6: "Fin límite 80 km/h", 7: "Límite 100 km/h", 8: "Límite 120 km/h",
    9: "No adelantar", 10: "No adelantar pesados", 11: "Intersección prioridad",
    12: "Carretera prioridad", 13: "Ceder el paso", 14: "Stop",
    15: "No hay paso", 16: "Prohibido pesados", 17: "Dirección prohibida",
    18: "Peligro general", 19: "Curva izquierda", 20: "Curva derecha",
    21: "Curvas peligrosas", 22: "Pavimento irregular", 23: "Calzada deslizante",
    24: "Estrechamiento derecha", 25: "Obras", 26: "Semáforo",
    27: "Peatones", 28: "Niños", 29: "Ciclistas",
    30: "Nieve o hielo", 31: "Animales salvajes", 32: "Fin restricciones",
    33: "Girar derecha", 34: "Girar izquierda", 35: "Seguir recto",
    36: "Recto o derecha", 37: "Recto o izquierda", 38: "Mantener derecha",
    39: "Mantener izquierda", 40: "Rotonda", 41: "Fin no adelantar",
    42: "Fin no adelantar pesados"
}

st.title("🚦 Reconocimiento de Señales de Tránsito")
st.write("Sube una imagen de una señal de tránsito y el modelo la clasificará.")

archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen subida", width=200)

    img = np.array(imagen)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)

    pred = model.predict(img_input)
    clase = np.argmax(pred)
    confianza = np.max(pred) * 100

    st.success(f"**Clase predicha:** {clase} — {nombres_clases.get(clase, 'Desconocida')}")
    st.info(f"**Confianza:** {confianza:.1f}%")