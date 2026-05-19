# Reconocimiento de Señales de Tránsito con CNN (GTSRB)

## Nombres del equipo
Manuela Alexandra Correa Gutierrez - 55222509
Yeimy Andrea Martínez Arévalo - 55222502
Andres Santiago Lanchers Ortiz - 55222522

## Reto 3
Desarrollar un modelo de visión por computador capaz de clasificar señales de tránsito en 43 categorías utilizando el dataset GTSRB, como parte de un sistema básico de asistencia a la conducción.

## Instrucciones para reproducir el experimento

Instalar dependencias:
pip install -r requirements.txt

Ejecutar la aplicación:
streamlit run app.py

## Dataset
GTSRB (German Traffic Sign Recognition Benchmark), dataset de clasificación multiclase de imágenes con 43 clases de señales de tránsito.

## Modelo
Red neuronal convolucional (CNN) entrenada en TensorFlow/Keras, con arquitectura basada en capas Conv2D, MaxPooling, Dropout y Dense.  
Entrada: imágenes de 32x32 píxeles  
Salida: 43 clases con activación softmax