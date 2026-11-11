import streamlit as st

from utils import load_image

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

    st.warning("Pipeline en construcción. Los módulos se van agregando por hito.")


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