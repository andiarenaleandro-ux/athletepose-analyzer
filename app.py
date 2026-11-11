import streamlit as st
import pandas as pd

from utils import load_image
from pose import load_pose_model, detect_pose, draw_skeleton


# ──────────────────────────────────────────────
# Modelo cacheado
# ──────────────────────────────────────────────
@st.cache_resource
def get_pose_model():
    """Carga MoveNet una sola vez y lo mantiene en memoria entre interacciones."""
    return load_pose_model()


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────
def tab_pipeline():
    st.header("Pose Analyzer — Detección articular")
    st.markdown(
        "Herramienta de análisis postural basada en **MoveNet** (Google / TensorFlow Hub). "
        "Detecta 17 keypoints articulares (formato COCO) en una foto de atleta."
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

    # ──────────────────────────────────────────
    # Detección de pose
    # ──────────────────────────────────────────
    st.divider()
    st.subheader("Detección de keypoints (MoveNet)")

    col_params, _ = st.columns([1, 2])
    with col_params:
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.1, max_value=0.9, value=0.3, step=0.05,
            help="Keypoints con confianza menor se marcan como no detectados.",
        )

    with st.spinner("Detectando pose con MoveNet..."):
        model = get_pose_model()
        keypoints = detect_pose(image, model, confidence_threshold=confidence_threshold)
        image_with_skeleton = draw_skeleton(image, keypoints)

    # Vista comparativa original vs con skeleton.
    col_orig, col_skel = st.columns(2)
    with col_orig:
        st.image(image, caption="Original", use_container_width=True)
    with col_skel:
        st.image(
            image_with_skeleton,
            caption="Con esqueleto (puntos: azul = izq atleta / der imagen)",
            use_container_width=True,
        )

    # Métrica de calidad de detección.
    detected_count = sum(1 for kp in keypoints if kp["detected"])
    st.metric(
        label="Keypoints detectados",
        value=f"{detected_count} / 17",
        delta=f"{detected_count / 17 * 100:.0f}% de cobertura",
        delta_color="off",
    )

    # Mensaje interpretativo según cobertura.
    if detected_count >= 15:
        st.success("✅ Detección completa — pose bien definida.")
    elif detected_count >= 10:
        st.info(
            "ℹ️ Detección parcial — algunos keypoints no se detectaron. "
            "Puede deberse a oclusión, baja resolución o pose fuera del encuadre."
        )
    else:
        st.warning(
            "⚠️ Detección pobre — la foto puede no ser óptima. "
            "Probá con una imagen con el atleta frontal, iluminación clara y encuadre completo."
        )

    # Tabla detallada de keypoints (expandible).
    with st.expander("Ver detalle de los 17 keypoints"):
        df = pd.DataFrame(keypoints)
        df["confidence"] = df["confidence"].map(lambda v: f"{v:.2f}")
        df["detected"] = df["detected"].map(lambda v: "✅" if v else "❌")
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.warning(
        "🚧 En construcción: próximos módulos de clasificación del deporte "
        "(MobileNetV2) y cálculo de ángulos articulares con interpretación automática."
    )


def main():
    st.set_page_config(
        page_title="AthletePose Analyzer — Parcial 2 PI",
        layout="wide",
    )
    st.title("AthletePose Analyzer")
    st.caption("Leandro Andiarena — Procesamiento de Imágenes — 2026")

    tab_pipeline()


if __name__ == "__main__":
    main()