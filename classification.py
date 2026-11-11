import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)


def load_classifier():
    """
    Carga el modelo MobileNetV2 preentrenado en ImageNet.

    La primera llamada descarga los pesos (~14 MB) a ~/.keras/models/
    y los deja cacheados. Las siguientes son instantáneas.

    Returns:
        model: instancia de MobileNetV2 lista para inferencia.
    """
    model = MobileNetV2(weights="imagenet")
    return model


def classify_image(
    image_rgb: np.ndarray,
    model,
    top_k: int = 3,
) -> list[tuple[str, str, float]]:
    """
    Clasifica una imagen RGB usando MobileNetV2 preentrenado en ImageNet.

    Args:
        image_rgb: imagen RGB (uint8), shape (H, W, 3)
        model: instancia de MobileNetV2 (devuelta por load_classifier())
        top_k: cantidad de predicciones top a devolver

    Returns:
        predictions: lista de tuplas (imagenet_id, label, confidence)
                     ordenadas de mayor a menor confianza.
                     Ejemplo: [('n03980874', 'poncho', 0.82),
                               ('n04254777', 'sock', 0.07), ...]
    """
    # 1. Resize a 224x224 (input obligatorio de MobileNetV2).
    #    cv2.resize usa interpolación bilineal por default (suficiente para CNN).
    resized = cv2.resize(image_rgb, (224, 224))

    # 2. Convertir a float32 (requerido por TensorFlow para operaciones numéricas).
    resized = resized.astype(np.float32)

    # 3. Agregar dimensión de batch: (224,224,3) → (1,224,224,3).
    #    TF siempre espera un batch, aunque tengamos una sola imagen.
    batch = np.expand_dims(resized, axis=0)

    # 4. Preprocesamiento específico de MobileNetV2:
    #    escala [0, 255] → [-1, 1] (la red fue entrenada con ese rango).
    batch = preprocess_input(batch)

    # 5. Inferencia. verbose=0 silencia los logs de progreso de TF.
    predictions = model.predict(batch, verbose=0)

    # 6. Decodificar: convertir vector de 1000 probabilidades en top-K predicciones
    #    con labels legibles. decode_predictions devuelve una lista por imagen del batch;
    #    como pasamos 1 imagen, tomamos [0].
    decoded = decode_predictions(predictions, top=top_k)[0]

    return decoded