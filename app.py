import streamlit as st
import pandas as pd

from utils import load_image
from pose import load_pose_model, detect_pose, draw_skeleton
from classification import load_classifier, classify_image
from metrics import (calculate_joint_angles, classify_sport_category,interpret_angles,)



# Modelos cacheados
@st.cache_resource
def get_pose_model():
    """Carga MoveNet una sola vez y lo mantiene en memoria entre interacciones."""
    return load_pose_model()

@st.cache_resource
def get_classifier():
    """Carga MobileNetV2 una sola vez."""
    return load_classifier()

# Pipeline principal
def tab_pipeline():
    st.header("Pose Analyzer — Detección articular")
    st.markdown(
        "Herramienta de análisis postural basada en **MoveNet** (Google / TensorFlow Hub). "
        "Detecta 17 keypoints articulares (formato COCO) en una foto de atleta, "
        "clasifica el deporte con **MobileNetV2** y evalúa los ángulos contra rangos de referencia."
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


    # 1. Detección de pose (MoveNet)

    st.divider()
    st.subheader("1. Detección de keypoints (MoveNet)")

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

    col_orig, col_skel = st.columns(2)
    with col_orig:
        st.image(image, caption="Original", use_container_width=True)
    with col_skel:
        st.image(
            image_with_skeleton,
            caption="Con esqueleto (puntos: azul = izq atleta / der imagen)",
            use_container_width=True,
        )

    detected_count = sum(1 for kp in keypoints if kp["detected"])
    st.metric(
        label="Keypoints detectados",
        value=f"{detected_count} / 17",
        delta=f"{detected_count / 17 * 100:.0f}% de cobertura",
        delta_color="off",
    )

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

    with st.expander("Ver detalle de los 17 keypoints"):
        df = pd.DataFrame(keypoints)
        df["confidence"] = df["confidence"].map(lambda v: f"{v:.2f}")
        df["detected"] = df["detected"].map(lambda v: "✅" if v else "❌")
        st.dataframe(df, use_container_width=True, hide_index=True)


    # 2. Cálculo de ángulos articulares
    # ──────────────────────────────────────────
    st.divider()
    st.subheader("2. Ángulos articulares calculados")
    st.markdown(
        "Ángulos medidos sobre los keypoints detectados. "
        "Cada ángulo requiere que los tres keypoints involucrados estén detectados; "
        "si falta alguno, el ángulo no se calcula."
    )

    angles = calculate_joint_angles(keypoints)

    joint_rows = [
        ("rodilla_izq", "rodilla_der"),
        ("codo_izq",    "codo_der"),
        ("cadera_izq",  "cadera_der"),
    ]

    for row in joint_rows:
        cols = st.columns(2)
        for col, joint_name in zip(cols, row):
            with col:
                data = angles[joint_name]
                label = joint_name.replace("_", " ").title()
                if data["available"]:
                    st.metric(
                        label=label,
                        value=f"{data['angle']:.1f}°",
                        delta="detectado",
                        delta_color="off",
                    )
                else:
                    st.metric(
                        label=label,
                        value="N/A",
                        delta="keypoints faltantes",
                        delta_color="inverse",
                    )

    available_count = sum(1 for a in angles.values() if a["available"])
    total = len(angles)
    st.caption(
        f"Ángulos calculados: **{available_count} de {total}**. "
        "Los ángulos faltantes se deben a keypoints no detectados "
        "(por oclusión, baja confianza o encuadre parcial)."
    )

    with st.expander("Detalle técnico de los ángulos"):
        rows = []
        for joint_name, data in angles.items():
            rows.append({
                "Articulación": joint_name.replace("_", " ").title(),
                "Ángulo (°)": f"{data['angle']:.1f}" if data["available"] else "N/A",
                "Keypoints usados": " → ".join(data["keypoints_used"]),
                "Disponible": "✅" if data["available"] else "❌",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


    # 3. Clasificación del deporte (MobileNetV2)
    # ──────────────────────────────────────────
    st.divider()
    st.subheader("3. Clasificación del deporte (MobileNetV2)")
    st.markdown(
        "Red convolucional preentrenada en ImageNet. Clasifica el contenido "
        "de la imagen; mapeamos las predicciones a categorías deportivas para "
        "seleccionar rangos de referencia de articulaciones."
    )

    with st.spinner("Clasificando con MobileNetV2..."):
        classifier = get_classifier()
        predictions = classify_image(image, classifier, top_k=3)

    cols = st.columns(3)
    for i, (imagenet_id, label, confidence) in enumerate(predictions):
        with cols[i]:
            st.metric(
                label=f"Top {i + 1}",
                value=label.replace("_", " ").title(),
                delta=f"{confidence * 100:.1f}%",
                delta_color="off",
            )

    category, trigger = classify_sport_category(predictions)
    if trigger:
        st.success(
            f"Categoría deportiva inferida: **{category}** "
            f"(disparada por '{trigger}')"
        )
    else:
        st.info(
            "ℹ️ Categoría deportiva inferida: **General** "
            "(ninguna clase de ImageNet matcheó con un deporte conocido)."
        )


    # 4. Interpretación automática según deporte
    # ──────────────────────────────────────────
    st.divider()
    st.subheader("4. Interpretación según categoría deportiva")

    interpretations = interpret_angles(angles, category)

    if not interpretations:
        st.info(
            "No hay interpretación automática disponible (categoría 'General' "
            "o sin rangos definidos)."
        )
    else:
        in_range = sum(1 for r in interpretations if r["status"] == "in_range")
        total_interp = len(interpretations)
        st.markdown(
            f"**Resumen:** {in_range} de {total_interp} ángulos dentro del rango esperado."
        )

        for item in interpretations:
            joint_label = item["joint"].replace("_", " ").title()
            angle_str = f"{item['angle']:.1f}°"

            if item["status"] == "in_range":
                st.success(f"✅ **{joint_label}: {angle_str}** — {item['message']}")
            elif item["status"] == "mild_deviation":
                st.warning(f"⚠️ **{joint_label}: {angle_str}** — {item['message']}")
            else:
                st.error(f"❌ **{joint_label}: {angle_str}** — {item['message']}")


# Main
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AthletePose Analyzer",
        layout="wide",
    )
    st.title("AthletePose Analyzer")
    st.caption("Leandro Andiarena — Procesamiento de Imágenes — 2026")

    tab_pipeline()


if __name__ == "__main__":
    main()