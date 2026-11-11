import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def load_image(uploaded_file):
    
    """Convierte un UploadedFile de Streamlit a un array NumPy RGB."""
    
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    uploaded_file.seek(0)  # resetear para posibles re-lecturas
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("No se pudo decodificar la imagen.")
        return None
    
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_histogram(image, title="Histograma", ax=None):
    
    """Dibuja histograma RGB de una imagen en un eje matplotlib."""
    
    if ax is None:
        fig, ax = plt.subplots()
        
    colors = ("red", "green", "blue")
    
    for i, c in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=c, linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    
    return ax