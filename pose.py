import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2


# Constantes del modelo MoveNet SinglePose Lightning
# ──────────────────────────────────────────────
MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
INPUT_SIZE = 192  # Lightning usa inputs de 192x192 píxeles

# Nombres de los 17 keypoints en orden COCO (estándar universal).
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Pares de keypoints que forman líneas del esqueleto para visualización (COCO).
# Cada tupla es (índice_1, índice_2). Se usará en H2 para dibujar el skeleton.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Cabeza
    (5, 6),                                # Hombros
    (5, 7), (7, 9), (6, 8), (8, 10),      # Brazos
    (5, 11), (6, 12), (11, 12),           # Tronco
    (11, 13), (13, 15), (12, 14), (14, 16),  # Piernas
]


def load_pose_model():
    """
    Carga MoveNet SinglePose Lightning desde TensorFlow Hub.

    Primera llamada: descarga ~10MB a un cache local del sistema.
    Llamadas siguientes: instantáneas.

    Returns:
        movenet: función invocable del modelo (signature 'serving_default')
    """
    model = hub.load(MOVENET_URL)
    movenet = model.signatures['serving_default']
    return movenet


def detect_pose(
    image_rgb: np.ndarray,
    model,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """
    Detecta los 17 keypoints articulares de una persona en la imagen.

    Usa MoveNet SinglePose: detecta UNA persona (la más dominante).
    Las coordenadas se devuelven en píxeles de la imagen ORIGINAL.

    Args:
        image_rgb: imagen RGB (uint8), shape (H, W, 3)
        model: modelo MoveNet cargado con load_pose_model()
        confidence_threshold: umbral mínimo de confianza (0.0 - 1.0).
                              Keypoints con score menor se marcan como no detectados.

    Returns:
        Lista de 17 dicts, uno por keypoint en orden COCO. Cada dict tiene:
            - name (str): nombre del keypoint (ej. "left_knee")
            - x (int): coordenada X en píxeles de la imagen original
            - y (int): coordenada Y en píxeles de la imagen original
            - confidence (float): score de confianza [0, 1]
            - detected (bool): True si confidence >= threshold
    """
    H, W = image_rgb.shape[:2]

    #  Preparar input: resize a INPUT_SIZE x INPUT_SIZE con padding
    #    para mantener aspect ratio, agregar dimensión de batch, int32.
    input_image = tf.image.resize_with_pad(
        tf.expand_dims(image_rgb, axis=0),
        INPUT_SIZE, INPUT_SIZE,
    )
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Inferencia. outputs['output_0'] tiene shape (1, 1, 17, 3).
    outputs = model(input_image)
    keypoints_raw = outputs['output_0'].numpy()[0, 0, :, :]  # shape (17, 3)

    # Calcular parámetros del padding usado por resize_with_pad.
    # Necesario para denormalizar correctamente las coordenadas.
    scale = min(INPUT_SIZE / H, INPUT_SIZE / W)
    scaled_H = H * scale
    scaled_W = W * scale
    pad_y = (INPUT_SIZE - scaled_H) / 2
    pad_x = (INPUT_SIZE - scaled_W) / 2

    # Convertir cada keypoint a coordenadas de píxel de la imagen original.
    keypoints = []
    for i, (y_norm, x_norm, confidence) in enumerate(keypoints_raw):
        # Pasar de normalizado [0,1] a píxel en espacio PADDED.
        y_padded = y_norm * INPUT_SIZE
        x_padded = x_norm * INPUT_SIZE

        # Remover el padding para obtener píxel en espacio ESCALADO.
        y_scaled = y_padded - pad_y
        x_scaled = x_padded - pad_x

        # Re-escalar a tamaño original.
        y_px = int(y_scaled / scale)
        x_px = int(x_scaled / scale)

        keypoints.append({
            "name": KEYPOINT_NAMES[i],
            "x": x_px,
            "y": y_px,
            "confidence": float(confidence),
            "detected": float(confidence) >= confidence_threshold,
        })

    return keypoints

def draw_skeleton(
    image_rgb: np.ndarray,
    keypoints: list[dict],
    point_radius: int = 5,
    line_thickness: int = 2,
    point_color: tuple = (255, 0, 0),
    line_color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Dibuja el esqueleto (keypoints + líneas) sobre una copia de la imagen.

    Solo se dibujan keypoints con detected=True.
    Las líneas se dibujan solo si AMBOS extremos están detectados.

    Args:
        image_rgb: imagen RGB (uint8), shape (H, W, 3)
        keypoints: lista de 17 dicts devuelta por detect_pose()
        point_radius: radio de los círculos de keypoints (px)
        line_thickness: grosor de las líneas del esqueleto (px)
        point_color: color RGB de los puntos
        line_color: color RGB de las líneas

    Returns:
        Imagen RGB (uint8) con el esqueleto dibujado.
    """
    output = image_rgb.copy()

    # Dibujar líneas entre pares de keypoints (según SKELETON_EDGES).
    for (idx_a, idx_b) in SKELETON_EDGES:
        kp_a = keypoints[idx_a]
        kp_b = keypoints[idx_b]
        if kp_a["detected"] and kp_b["detected"]:
            cv2.line(
                output,
                (kp_a["x"], kp_a["y"]),
                (kp_b["x"], kp_b["y"]),
                line_color,
                line_thickness,
            )

    # Dibujar círculos en cada keypoint detectado.
    for kp in keypoints:
        if kp["detected"]:
            cv2.circle(
                output,
                (kp["x"], kp["y"]),
                point_radius,
                point_color,
                -1,  # relleno
            )

    return output