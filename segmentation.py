import cv2
import numpy as np


def segment_kmeans(
    image_rgb: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    epsilon: float = 0.2,
    attempts: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segmenta una imagen RGB usando K-means clustering sobre el espacio de color.

    Agrupa los píxeles en K clusters por similitud de color. Cada píxel queda
    asignado al cluster cuyo centro (color promedio) es más cercano en
    distancia euclidiana.

    Args:
        image_rgb: imagen RGB (uint8), shape (H, W, 3)
        k: cantidad de clusters (típicamente 2-6)
        max_iter: iteraciones máximas del algoritmo
        epsilon: tolerancia de convergencia (cambio mínimo de centros)
        attempts: número de ejecuciones con distintas inicializaciones;
                  se devuelve el mejor resultado (menor compactness).

    Returns:
        segmented: imagen reconstruida donde cada píxel toma el color del
                   centro de su cluster (uint8, shape (H, W, 3))
        labels: array (H, W) con el cluster asignado a cada píxel (0..k-1)
        centers: colores promedio de cada cluster (uint8, shape (k, 3))
    """
    h, w = image_rgb.shape[:2]

    # 1. Reshape: (H, W, 3) → (N, 3) donde N = H*W.
    #    K-means trabaja sobre vectores de features (filas), no matrices 2D.
    data = image_rgb.reshape((-1, 3)).astype(np.float32)

    # 2. Criterios de parada: termina por epsilon O por max_iter (lo que ocurra primero).
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon,
    )

    # 3. Ejecutar K-means con K-means++ (inicialización inteligente).
    _, labels_flat, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    # 4. Centers vienen como float32 (promedios). Para visualizar, uint8.
    centers = np.uint8(centers)

    # 5. Reconstruir imagen segmentada: cada píxel toma el color del centro
    #    de su cluster. labels_flat es un array (N, 1); flatten lo pasa a (N,).
    segmented_flat = centers[labels_flat.flatten()]
    segmented = segmented_flat.reshape(image_rgb.shape)

    # 6. Reshape labels a (H, W) para alinear con la imagen espacialmente.
    labels = labels_flat.reshape((h, w))

    return segmented, labels, centers


def build_mask_from_clusters(
    labels: np.ndarray,
    selected_clusters: list[int],
) -> np.ndarray:
    """
    Construye una máscara binaria a partir de los clusters seleccionados.

    Args:
        labels: array (H, W) con el cluster asignado a cada píxel
        selected_clusters: lista de IDs de clusters a incluir en la máscara

    Returns:
        mask: array (H, W) uint8 con valores 0 (fondo) y 255 (atleta/objeto)
    """
    # np.isin devuelve True donde el label está en la lista de seleccionados.
    # .astype(uint8) * 255 lo convierte a formato de imagen binaria.
    mask = np.isin(labels, selected_clusters).astype(np.uint8) * 255
    return mask