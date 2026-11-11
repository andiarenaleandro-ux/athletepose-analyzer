# AthletePose Analyzer — Parcial 2 Procesamiento de Imágenes

**Leandro Andiarena** ·Ciencia de Datos e IA · ISSD · 4° Semestre · 2026

---

## 1. Descripción del programa

**AthletePose Analyzer** es una herramienta de caracterización postural automática para fotos de atletas. Dada una imagen estática de un deportista, el programa:

1. **Detecta 17 puntos articulares** (keypoints COCO) del cuerpo del atleta.
2. **Dibuja el esqueleto** sobre la imagen (puntos + líneas).
3. **Calcula 6 ángulos articulares** (rodillas, codos y caderas).
4. **Clasifica el deporte/actividad** mediante una CNN preentrenada.
5. **Interpreta automáticamente** los ángulos contra rangos de referencia típicos del deporte identificado.

El resultado es un reporte visual + numérico útil como pre-procesador para análisis deportivo, kinesiología, ergonomía o como fase 0 de un sistema más completo (ej. análisis de video con pose estimation temporal).

---

## 2. Enfoque y motivación

Se tomaron dos decisiones estratégicas:

**a)**
El pipeline usa **MobileNetV2** (CNN preentrenada) como componente de clasificación, integrada con **MoveNet** (Google / TensorFlow Hub) como motor de pose estimation. MoveNet no fue visto en clase: su incorporación apunta a la investigacion de modelos preentrenados hacia una tarea más específica.

**b)**
El programa se limita a **análisis de foto estática** (no video, no pose temporal).

---

## 3. Arquitectura del proyecto

Estructura modular con separación de responsabilidades: los módulos de procesamiento no dependen de Streamlit, son funciones puras reutilizables.

AthletePose Analyzer 
        ├── app.py # UI Streamlit + orquestación del pipeline
        ├── pose.py # MoveNet: carga del modelo, detección, skeleton
        ├── classification.py # MobileNetV2: carga y clasificación
        ├── metrics.py # Ángulos articulares + clasificación de categoría + interpretación
        ├── utils.py # load_image
        ├── requirements.txt # Dependencias con versiones pineadas
        ├── explicacion.md # Este documento
        ├── README.md # Setup y ejecución
        └── imagenes/ # 4 imágenes de ejemplo


---

## 4. Técnicas utilizadas

| Técnica | Origen | Rol en el pipeline |
|---|---|---|
| **MobileNetV2** | Keras / TensorFlow |Clasifica el contenido de la imagen (1000 clases ImageNet) → mapea a categoría deportiva |
| **MoveNet SinglePose Lightning** | Google / TensorFlow Hub | Detecta 17 keypoints articulares del cuerpo (formato COCO) |
| **Geometría vectorial (NumPy)** | NumPy puro | Cálculo de ángulos articulares vía producto escalar + arccos |
| **Streamlit** | Streamlit  | Interfaz web interactiva |

---

## 5. Pipeline del sistema
Imagen de entrada (RGB)
│
▼
┌─────────────────────────────┐
│ 1. Detección de keypoints │ ← MoveNet (TF Hub)
│ 17 puntos articulares │
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ 2. Cálculo de ángulos │ ← NumPy puro
│ 6 articulaciones clave │
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ 3. Clasificación de deporte │ ← MobileNetV2 (Keras)
│ Top-3 ImageNet │
│ → categoría funcional │
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ 4. Interpretación automática│ ← Reglas + rangos de referencia
│ Ángulos vs rangos típicos│
└─────────────────────────────┘
│
▼
Reporte final


---

### 5.1 Detección de keypoints (MoveNet)

MoveNet es una CNN de Google optimizada para pose estimation en CPU. Variante **Lightning** (~10 MB). Devuelve 17 keypoints en formato **COCO** (nariz, ojos, orejas, hombros, codos, muñecas, caderas, rodillas, tobillos) con sus coordenadas normalizadas y un score de confianza por keypoint.

Se incluye un **umbral de confianza configurable** (default 0.3): keypoints por debajo se marcan como "no detectados" y no participan en cálculos posteriores. Esto evita que detecciones poco confiables generen ángulos erróneos.

### 5.2 Cálculo de ángulos (geometría vectorial)

Para cada articulación de interés, se forman dos vectores desde el vértice hacia los extremos y se calcula el ángulo entre ellos:

v1 = p1 - p2 (vector del vértice al primer extremo)
v2 = p3 - p2 (vector del vértice al segundo extremo)
cos(θ) = (v1 · v2) / (|v1| · |v2|)
θ = arccos(cos(θ))


Las 6 articulaciones calculadas: **rodilla izq/der** (cadera-rodilla-tobillo), **codo izq/der** (hombro-codo-muñeca), **cadera izq/der** (hombro-cadera-rodilla).

Un ángulo solo se calcula si **los 3 keypoints involucrados fueron detectados**; de lo contrario se marca como `N/A`. Esta decisión evita devolver valores erróneos.

### 5.3 Clasificación de categoría deportiva (MobileNetV2)

MobileNetV2 preentrenada en ImageNet (1.28M imágenes, 1000 clases) genera las top-3 predicciones de la imagen. Luego se mapea cada label a una **categoría funcional**:

| Categoría | Clases ImageNet que la disparan |
|---|---|
| Weightlifting | dumbbell, barbell |
| Cycling | mountain_bike, unicycle, bicycle-built-for-two, tricycle |
| Running | running_shoe, jersey, sweatshirt, track |
| Yoga | balance_beam, leotard, maillot, tights, yoga_mat |
| **General (fallback)** | ninguna del listado |

Se aplica un **umbral de confianza mínima (10%)** para asignar categoría. Si ninguna predicción supera el umbral, el sistema cae a "General" y **no fuerza interpretación**.

### 5.4 Interpretación automática

Cada ángulo detectado se compara contra los rangos típicos de la categoría asignada. Se asigna un status:

- **In range** — dentro del rango esperado
- **Mild deviation** — fuera por menos de 15°
- **Severe deviation** — fuera por más de 15°

---

## 6. Decisiones de diseño clave

**6.1 Separación de responsabilidades.**
Cada módulo tiene función única. `pose.py` no dibuja en pantalla (solo devuelve keypoints), `metrics.py` no sabe de Streamlit (solo hace cálculos), `app.py` solo orquesta y renderiza. Permite testing independiente y reuso en otros contextos (API, notebook, script batch).

**6.2 Caché de modelos.**
Tanto MoveNet como MobileNetV2 se cargan una sola vez por sesión mediante `@st.cache_resource`. Sin esta cache, cada movimiento de un slider recargaría los modelos haciendo la app inusable.

**6.3 Fallback "General" explícito.**
Cuando el modelo no clasifica con confianza suficiente, el sistema admite la limitación y muestra solo los ángulos crudos, sin forzar una interpretación equivocada. 

**6.4 Mapeo ImageNet → categoría funcional.**
En lugar de intentar que MobileNetV2 clasifique deportes directamente (algo que no fue entrenado para hacer), se aprovecha lo que sí sabe hacer (identificar objetos) y se construye una capa de mapeo en Python. Es simple, transparente y modificable sin reentrenar nada.

**6.5 Umbral de confianza en keypoints.**
Aceptar todos los keypoints sin umbral generaría ángulos basados en detecciones poco confiables. Rechazarlos con un umbral mejora la calidad de los ángulos reportados.

**6.6 Nomenclatura "left/right" desde la perspectiva del atleta.**
Se respeta el estándar COCO: `left_shoulder` es el hombro izquierdo Del atleta, que en una foto frontal aparece a la derecha de la imagen. Es el estándar universal.

---

## 7. Casos de prueba documentados

Se probó el sistema con 4 imágenes representativas de distintos comportamientos del pipeline.

### Caso 1 — Weightlifting (sentadilla con barbell)



**Comportamiento:** clasificación exitosa como **Weightlifting** por detección de `barbell`. Pose detectada completa. Los ángulos reflejan correctamente la postura de sentadilla profunda (rodillas ~50°, cadera ~45°, codos ~25°).


### Caso 2 — Yoga (Warrior II)



**Comportamiento:** 17/17 keypoints detectados. Pose correctamente reconstruida. **Clasificación con baja confianza** (top-1: Balance Beam 8.2%, top-2: Dumbbell 5.7%). El umbral de confianza (10%) **rechaza asignar categoría** y cae a "General".

**Esto es diseñado**: ImageNet no tiene clases explícitas de yoga. El sistema reconoce la limitación y entrega los ángulos crudos sin interpretación forzada.

### Caso 3 — Running (sprint lateral)


**Comportamiento:** pose detectada correctamente. **Clasificación cae a "General"** porque el modelo asocia la figura a clases como `Rugby Ball` (19.6%) o `Racket` (11.9%) — un sesgo conocido de ImageNet donde objetos deportivos dominan sobre actividades humanas sin equipamiento visible.

El sistema maneja esto correctamente: muestra honestamente el top-3 con confianzas bajas, cae en "General" y no fuerza interpretación equivocada.

### Caso 4 — Cycling (posición time trial)

**Comportamiento:** **clasificación muy confiada** (Mountain Bike 95.7%) → categoría **Cycling** asignada. Los rangos definidos corresponden a posición de ruta clásica; esta foto muestra posición aero/TT, más agresiva, por lo que algunos ángulos pueden caer fuera de rango.

**Observación defendible:** dentro de "Cycling" existen sub-disciplinas (ruta, TT, MTB, pista) con rangos distintos. El sistema no distingue entre ellas — mejora futura prevista.

---

## 8. Limitaciones conocidas

**8.1 Dataset ImageNet no cubre todas las actividades.**
ImageNet tiene clases de objetos (dumbbell, mountain_bike, running_shoe) pero no de actividades (correr, nadar, hacer yoga). El mapeo a categorías funcionales es una aproximación. Actividades sin equipamiento distintivo (yoga, calistenia, running sin ropa deportiva) tienden a caer en "General".

**8.2 Rangos de referencia son aproximaciones.**
Los rangos definidos por categoría son orientativos, basados en literatura general de biomecánica deportiva. **No constituyen diagnóstico**. Dentro de cada deporte existen muchas posturas específicas con rangos propios (ej. sentadilla vs. deadlift vs. press militar).

**8.3 MoveNet SinglePose detecta una sola persona.**
Si la foto tiene múltiples personas, MoveNet detecta la más dominante. Para análisis de varias personas simultáneas haría falta MoveNet MultiPose o YOLOv8-Pose.

**8.4 Foto estática: no hay análisis temporal.**
El sistema analiza una instantánea. No puede evaluar la ejecución dinámica de un ejercicio (ej. simetría del pedaleo, velocidad de un sprint). Para análisis dinámico requeriría pipeline sobre video.

**8.5 Vocabulario COCO: left/right desde la perspectiva del atleta.**
Si la foto no es frontal/trasera clásica, la nomenclatura "left/right" puede ser contraintuitiva para el usuario casual. Es el estándar de la industria y se documenta en la UI.

---

## 9. Roadmap futuro

El proyecto está diseñado como **Fase 0** de un sistema de análisis postural más completo. Direcciones de evolución:

| Fase | Mejora | Habilitador técnico |
|---|---|---|
| 1 | Sub-categorías por disciplina (ej. Cycling → ruta / TT / MTB) con rangos propios | Clasificador secundario sobre la imagen |
| 2 | Análisis de video (series temporales de ángulos, detección de ciclos) | OpenCV video + agregación temporal |
| 3 | Pose estimation en 3D (profundidad real) | Modelos como MediaPipe 3D o VideoPose3D |
| 4 | Reporte técnico-biomecánico con alertas | Reglas por deporte curadas con especialistas |

---

## 10. Stack técnico

| Librería | Versión | Rol |
|---|---|---|
| `opencv-python` | 4.10.0.84 | Lectura y manipulación de imágenes |
| `numpy` | >=1.24 | Álgebra vectorial, cálculo de ángulos |
| `tensorflow` | >=2.13 | Backend de MobileNetV2 y MoveNet |
| `tensorflow-hub` | >=0.15 | Carga de MoveNet desde TF Hub |
| `streamlit` | >=1.28 | Interfaz web |
| `pandas` | (transitiva) | Tablas de keypoints y ángulos |
| `matplotlib` | >=3.7 | Visualizaciones auxiliares |
| `Pillow` | >=10.0 | Dependencia de Streamlit |
| `setuptools` | <80 | Fix de compatibilidad con tensorflow-hub |

---

## 11. Cómo ejecutar

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr la aplicación
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

**Primera corrida:** MoveNet (~10 MB) y MobileNetV2 (~14 MB) se descargan desde los repositorios oficiales. Puede tardar 20-60 segundos. Las siguientes corridas usan cache local y son instantáneas.

**Imágenes de ejemplo:** la carpeta `imagenes/` contiene 4 fotos de prueba cubriendo los distintos comportamientos del pipeline (ver sección 7).

---

## 12. Reflexión final

Este trabajo combina una técnica de MobileNetV2 como CNN preentrenada con MoveNet como modelo de pose estimation, aplicados en un pipeline: detectar postura, identificar actividad, evaluar ángulos.

---