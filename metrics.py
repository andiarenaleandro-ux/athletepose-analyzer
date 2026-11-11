import numpy as np


# Definición de articulaciones a medir.
# Cada entrada: (nombre, índice_p1, índice_p2_vértice, índice_p3)

JOINT_DEFINITIONS = [
    # (name, p1 (lejano), p2 (vértice), p3 (lejano))
    ("rodilla_izq",         11, 13, 15),  # cadera_izq - rodilla_izq - tobillo_izq
    ("rodilla_der",         12, 14, 16),  # cadera_der - rodilla_der - tobillo_der
    ("codo_izq",             5,  7,  9),  # hombro_izq - codo_izq - muñeca_izq
    ("codo_der",             6,  8, 10),  # hombro_der - codo_der - muñeca_der
    ("cadera_izq",           5, 11, 13),  # hombro_izq - cadera_izq - rodilla_izq
    ("cadera_der",           6, 12, 14),  # hombro_der - cadera_der - rodilla_der
]


def calculate_angle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    """
    Calcula el ángulo en grados formado en el vértice p2 por los puntos p1-p2-p3.

    Usa producto escalar + arccos. Rango de salida: 0° a 180°.

    Args:
        p1: primer extremo del ángulo (x, y)
        p2: vértice del ángulo (x, y)
        p3: segundo extremo del ángulo (x, y)

    Returns:
        Ángulo en grados (0.0 - 180.0)
    """
    # Convertir a arrays NumPy para operaciones vectoriales.
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)

    # Vectores desde el vértice hacia los extremos.
    v1 = p1 - p2
    v2 = p3 - p2

    # Normas (longitudes de los vectores).
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Guardia contra división por cero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # Coseno del ángulo. Clipeamos a [-1, 1] para evitar errores de arccos
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Arccos devuelve radianes; convertimos a grados.
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def calculate_joint_angles(keypoints: list[dict]) -> dict[str, dict]:
    """
    Calcula los ángulos de las articulaciones clave a partir de los keypoints.

    Solo calcula un ángulo si los TRES keypoints involucrados fueron detectados
    (detected=True). Si falta alguno, devuelve angle=None.

    Args:
        keypoints: lista de 17 dicts devuelta por detect_pose()

    Returns:
        Dict con las articulaciones como claves. Cada valor es un dict:
            {
                "angle": float | None,   # grados, o None si no se pudo calcular
                "available": bool,       # True si se calculó
                "keypoints_used": list   # nombres de los 3 keypoints usados
            }
    """
    angles = {}

    for joint_name, idx_p1, idx_p2, idx_p3 in JOINT_DEFINITIONS:
        kp1 = keypoints[idx_p1]
        kp2 = keypoints[idx_p2]
        kp3 = keypoints[idx_p3]

        # Solo calculamos si los tres keypoints están detectados.
        if kp1["detected"] and kp2["detected"] and kp3["detected"]:
            angle = calculate_angle(
                (kp1["x"], kp1["y"]),
                (kp2["x"], kp2["y"]),
                (kp3["x"], kp3["y"]),
            )
            angles[joint_name] = {
                "angle": angle,
                "available": True,
                "keypoints_used": [kp1["name"], kp2["name"], kp3["name"]],
            }
        else:
            angles[joint_name] = {
                "angle": None,
                "available": False,
                "keypoints_used": [kp1["name"], kp2["name"], kp3["name"]],
            }

    return angles


# Mapeo ImageNet: categoría deportiva + rangos de referencia
# ──────────────────────────────────────────────

# Mapeo: label de ImageNet  → categoría deportiva
IMAGENET_TO_CATEGORY = {
    # Weightlifting
    "dumbbell": "Weightlifting",
    "barbell": "Weightlifting",

    # Cycling
    "mountain_bike": "Cycling",
    "unicycle": "Cycling",
    "bicycle-built-for-two": "Cycling",
    "tricycle": "Cycling",

    # Running
    "running_shoe": "Running",
    "jersey": "Running",
    "sweatshirt": "Running",
    "track": "Running",
    
    # Yoga / flexibility
    "balance_beam": "Yoga",
    "leotard": "Yoga",
    "maillot": "Yoga",
    "tights": "Yoga",
    "yoga_mat": "Yoga",

    # Agregar mapeos según las predicciones típicas se observen.
}


# Rangos de referencia por categoría y articulación.
# Fuentes: literatura general de biomecánica deportiva.
# IMPORTANTE: son orientativos
REFERENCE_RANGES = {
    "Weightlifting": {
        "rodilla_izq":  (40, 180),
        "rodilla_der":  (40, 180),
        "codo_izq":     (20, 180),
        "codo_der":     (20, 180),
        "cadera_izq":   (30, 180),
        "cadera_der":   (30, 180),
    },
    "Cycling": {
        "rodilla_izq":  (65, 150),
        "rodilla_der":  (65, 150),
        "codo_izq":     (140, 170),
        "codo_der":     (140, 170),
        "cadera_izq":   (40, 80),
        "cadera_der":   (40, 80),
    },
    "Running": {
        "rodilla_izq":  (110, 170),
        "rodilla_der":  (110, 170),
        "codo_izq":     (80, 120),
        "codo_der":     (80, 120),
        "cadera_izq":   (100, 160),
        "cadera_der":   (100, 160),
    },
    "Yoga": {
        # Rangos muy amplios porque las posturas de yoga varían enormemente.
        "rodilla_izq":  (60, 180),
        "rodilla_der":  (60, 180),
        "codo_izq":     (60, 180),
        "codo_der":     (60, 180),
        "cadera_izq":   (60, 180),
        "cadera_der":   (60, 180),
    },
}


def classify_sport_category(predictions: list[tuple[str, str, float]],min_confidence: float = 0.10,) -> tuple[str, str]:
    """
    Mapea las top-K predicciones de MobileNetV2 a una categoría deportiva.

    Toma la primera predicción cuya clase matchea IMAGENET_TO_CATEGORY
    Y supera el umbral de confianza mínima. Si ninguna califica,
    devuelve "General".

    Args:
        predictions: lista de tuplas (imagenet_id, label, confidence)
                     devueltas por classify_image() de MobileNetV2.
        min_confidence: umbral mínimo (0.0-1.0) para aceptar un match.
                        Default 0.10 (10%). Con valores menores, el modelo
                        no está seguro de nada y asignar categoría es riesgoso.

    Returns:
        (category, trigger_label):
            - category: "Weightlifting" | "Cycling" | "Running" | "Yoga" | "General"
            - trigger_label: la clase ImageNet que disparó el match,
                             o "" si cayó en fallback.
    """
    for imagenet_id, label, confidence in predictions:
        label_lower = label.lower()
        if label_lower in IMAGENET_TO_CATEGORY and confidence >= min_confidence:
            return IMAGENET_TO_CATEGORY[label_lower], label_lower
    return "General", ""

def interpret_angles(
    angles: dict[str, dict],
    category: str,
    tolerance_deg: float = 15.0,
) -> list[dict]:
    """
    Interpreta cada ángulo articular contra los rangos de referencia del deporte.

    Args:
        angles: dict devuelto por calculate_joint_angles()
        category: categoría deportiva (de classify_sport_category())
        tolerance_deg: cuántos grados fuera del rango se consideran "leve"

    Returns:
        Lista de dicts, uno por articulación evaluable:
            {
                "joint": str,
                "angle": float,
                "range": (min, max),
                "status": "in_range" | "mild_deviation" | "severe_deviation",
                "message": str  # texto legible
            }
        Si la categoría es "General", devuelve lista vacía (sin interpretación).
    """
    if category not in REFERENCE_RANGES:
        return []

    reference = REFERENCE_RANGES[category]
    results = []

    for joint_name, data in angles.items():
        if not data["available"]:
            continue  # saltamos los ángulos no disponibles

        angle = data["angle"]
        range_min, range_max = reference[joint_name]

        if range_min <= angle <= range_max:
            status = "in_range"
            message = f"Dentro del rango esperado ({range_min}°-{range_max}°)"
        elif angle < range_min - tolerance_deg or angle > range_max + tolerance_deg:
            deviation = max(range_min - angle, angle - range_max)
            status = "severe_deviation"
            message = (
                f"Significativamente fuera ({range_min}°-{range_max}°, "
                f"desviación {deviation:.0f}°)"
            )
        else:
            deviation = max(range_min - angle, angle - range_max)
            status = "mild_deviation"
            message = (
                f"Levemente fuera ({range_min}°-{range_max}°, "
                f"desviación {deviation:.0f}°)"
            )

        results.append({
            "joint": joint_name,
            "angle": angle,
            "range": (range_min, range_max),
            "status": status,
            "message": message,
        })

    return results