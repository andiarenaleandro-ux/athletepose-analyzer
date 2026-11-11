import cv2
import streamlit as st

from utils import load_image
from detection import detect_faces
from classification import load_classifier, classify_image


# ──────────────────────────────────────────────
# Modelos cacheados (se cargan una sola vez)
# ──────────────────────────────────────────────
@st.cache_resource
def get_classifier():
    """Carga MobileNetV2 una sola vez y lo cachea en memoria."""
    return load_classifier()


# ──────────────────────────────────────────────
# TAB 1 — Pipeline de caracterización de atleta
# ──────────────────────────────────────────────
def tab_pipeline():
    st.header("Caracterización de silueta deportiva")
    st.markdown(
        "Pipeline que detecta rostro, clasifica el deporte, segmenta al atleta "
        "del fondo, limpia la máscara, detecta puntos característicos y calcula "
        "métricas geométricas sobre la silueta."
    )

    uploaded = st.file_uploader(
        "Subí una foto de un atleta",
        type=["jpg", "jpeg", "png"],
        key="athlete_upload",
    )

    if not uploaded:
        st.info("Esperando imagen...")
        return

    image = load_image(uploaded)
    if image is None:
        return

    st.image(image, caption="Imagen original", use_container_width=True)
    st.divider()

    # ──────────────────────────────────────────
    # PASO 1 — Validación de encuadre (Haar)
    # ──────────────────────────────────────────
    st.subheader("1. Validación de encuadre (Haar Cascade)")
    st.markdown(
        "Detecta rostros frontales como paso de validación: confirma que la "
        "foto contiene al menos un atleta visible frontalmente."
    )

    col_params, col_result = st.columns([1, 2])

    with col_params:
        st.markdown("**Parámetros del detector**")
        scale_factor = st.slider(
            "Scale factor", 1.05, 1.40, 1.10, 0.05,
            help="Factor de reducción entre escalas. Menor = más preciso pero más lento.",
        )
        min_neighbors = st.slider(
            "Min neighbors", 1, 10, 5,
            help="Detecciones vecinas mínimas para validar un rostro.",
        )
        min_size_ratio = st.slider(
            "Min size ratio", 0.01, 0.20, 0.05, 0.01,
            help="Tamaño mínimo del rostro como fracción del lado corto.",
        )

    faces, gray = detect_faces(
        image_rgb=image,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size_ratio=min_size_ratio,
    )

    image_with_faces = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(image_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 3)

    with col_result:
        st.image(
            image_with_faces,
            caption=f"Rostros detectados: {len(faces)}",
            use_container_width=True,
        )

    if len(faces) == 0:
        st.warning(
            "⚠️ No se detectó ningún rostro. La foto puede ser lateral, a distancia, "
            "o el atleta puede estar de espaldas. El pipeline continúa pero sin "
            "validación de encuadre."
        )
    elif len(faces) > 1:
        st.info(
            f"ℹ️ Se detectaron {len(faces)} rostros. Para análisis individual, "
            "considerá usar una foto con un solo atleta."
        )
    else:
        st.success("✅ Encuadre validado: 1 rostro detectado.")

    st.divider()

    # ──────────────────────────────────────────
    # PASO 2 — Clasificación del deporte (MobileNetV2)
    # ──────────────────────────────────────────
    st.subheader("2. Clasificación del deporte (MobileNetV2)")
    st.markdown(
        "Red neuronal convolucional preentrenada en ImageNet (1.28M imágenes, "
        "1000 clases). Identifica el objeto o actividad principal de la imagen "
        "sin entrenamiento específico de deportes."
    )

    top_k = st.slider(
        "Cantidad de predicciones a mostrar (top-K)",
        min_value=1, max_value=5, value=3,
    )

    with st.spinner("Clasificando con MobileNetV2..."):
        classifier = get_classifier()
        predictions = classify_image(image, classifier, top_k=top_k)

    # Mostrar predicciones como métricas en columnas.
    cols = st.columns(top_k)
    for i, (imagenet_id, label, confidence) in enumerate(predictions):
        with cols[i]:
            st.metric(
                label=f"Top {i + 1}",
                value=label.replace("_", " ").title(),
                delta=f"{confidence * 100:.1f}%",
                delta_color="off",
            )

    # Nota interpretativa.
    top_label = predictions[0][1].replace("_", " ")
    top_conf = predictions[0][2] * 100
    if top_conf > 50:
        st.success(f"✅ Predicción fuerte: **{top_label}** ({top_conf:.1f}%).")
    elif top_conf > 20:
        st.info(f"ℹ️ Predicción moderada: **{top_label}** ({top_conf:.1f}%). "
                "El modelo no está del todo seguro — revisá el top-K completo.")
    else:
        st.warning(f"⚠️ Predicción débil: **{top_label}** ({top_conf:.1f}%). "
                   "La imagen puede estar fuera del dominio de ImageNet.")

    st.divider()

    # ──────────────────────────────────────────
    # PASOS 3 a 6 — Pendientes (próximos hitos)
    # ──────────────────────────────────────────
    st.warning(
        "Pipeline en construcción. Próximos módulos: segmentación del atleta, "
        "limpieza morfológica, detección de puntos característicos, métricas geométricas."
    )


def main():
    st.set_page_config(
        page_title="AthleteSilhouette — Parcial 2 PI",
        layout="wide",
    )
    st.title("AthleteSilhouette")
    st.caption("Leandro Andiarena — Procesamiento de Imágenes — 2026")

    tab1, = st.tabs(["Pipeline"])

    with tab1:
        tab_pipeline()


if __name__ == "__main__":
    main()