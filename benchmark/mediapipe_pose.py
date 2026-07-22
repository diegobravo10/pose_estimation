#!/usr/bin/env python3
"""
Benchmark de estimación de pose con MediaPipe Pose Landmarker Lite.

Dependencias (activar entorno virtual primero):
    python -m venv venv
    source venv/bin/activate
    pip install mediapipe opencv-python psutil

Uso:
    python benchmark/mediapipe_pose.py 0                     # webcam por defecto
    python benchmark/mediapipe_pose.py 2                     # webcam USB índice 2
    python benchmark/mediapipe_pose.py video/video1.mp4      # archivo de video
"""

import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import psutil
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────────────────────────────────────
# Rutas (siempre relativas a la raíz del proyecto, sin importar desde dónde
# se ejecute el script)
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

MODEL_PATH   = os.path.join(_PROJECT_ROOT, "model", "pose_landmarker_full.task")
CSV_PATH     = os.path.join(_PROJECT_ROOT, "results", "mediapipe_pose.csv")
WINDOW_TITLE = "MediaPipe Pose Landmarker FUll"

# ──────────────────────────────────────────────────────────────────────────────
# Conexiones del esqueleto de 33 landmarks de MediaPipe Pose.
# Equivalentes a mp.solutions.pose.POSE_CONNECTIONS, definidas explícitamente
# para no depender de la API legacy.
#
# Índices de landmarks:
#   0  NOSE          11 LEFT_SHOULDER   12 RIGHT_SHOULDER
#   1  LEFT_EYE_INNER  13 LEFT_ELBOW    14 RIGHT_ELBOW
#   2  LEFT_EYE        15 LEFT_WRIST    16 RIGHT_WRIST
#   3  LEFT_EYE_OUTER  17 LEFT_PINKY    18 RIGHT_PINKY
#   4  RIGHT_EYE_INNER 19 LEFT_INDEX    20 RIGHT_INDEX
#   5  RIGHT_EYE       21 LEFT_THUMB    22 RIGHT_THUMB
#   6  RIGHT_EYE_OUTER 23 LEFT_HIP      24 RIGHT_HIP
#   7  LEFT_EAR        25 LEFT_KNEE     26 RIGHT_KNEE
#   8  RIGHT_EAR       27 LEFT_ANKLE    28 RIGHT_ANKLE
#   9  MOUTH_LEFT      29 LEFT_HEEL     30 RIGHT_HEEL
#  10  MOUTH_RIGHT     31 LEFT_FOOT_IDX 32 RIGHT_FOOT_IDX
# ──────────────────────────────────────────────────────────────────────────────

POSE_CONNECTIONS: frozenset = frozenset([
    # Cara
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Hombros y torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Brazo izquierdo
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    # Brazo derecho
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    # Pierna izquierda
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Pierna derecha
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
])

# Landmarks con visibilidad por debajo de este umbral no se dibujan
VISIBILITY_THRESHOLD = 0.5

# Cada cuántos frames se hace flush del CSV (robustez ante cortes)
CSV_FLUSH_INTERVAL = 50

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def parse_source(arg: str):
    """
    Devuelve (is_webcam: bool, source: int | str).
    Si arg es convertible a entero, se trata como índice de cámara.
    """
    try:
        return True, int(arg)
    except ValueError:
        return False, arg


class FpsCounter:
    """
    Calcula FPS en tiempo real usando una ventana deslizante de timestamps.
    Evita el ruido de calcular frame-a-frame.
    """

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.monotonic())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0.0 else 0.0


def percentile(data: list, p: float) -> float:
    """Percentil p ∈ [0, 100] sobre una lista de floats."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Dibujo
# ──────────────────────────────────────────────────────────────────────────────

def draw_pose(frame, landmarks: list, height: int, width: int) -> None:
    """
    Dibuja conexiones del esqueleto y puntos de landmark sobre el frame.
    Operación in-place para evitar copias innecesarias.
    """
    # Conexiones primero (capa inferior)
    for idx_a, idx_b in POSE_CONNECTIONS:
        lm_a = landmarks[idx_a]
        lm_b = landmarks[idx_b]
        vis_a = lm_a.visibility if lm_a.visibility is not None else 0.0
        vis_b = lm_b.visibility if lm_b.visibility is not None else 0.0
        if vis_a < VISIBILITY_THRESHOLD or vis_b < VISIBILITY_THRESHOLD:
            continue
        x1 = int(lm_a.x * width)
        y1 = int(lm_a.y * height)
        x2 = int(lm_b.x * width)
        y2 = int(lm_b.y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)

    # Landmarks encima de las conexiones
    for lm in landmarks:
        vis = lm.visibility if lm.visibility is not None else 0.0
        if vis < VISIBILITY_THRESHOLD:
            continue
        cx = int(lm.x * width)
        cy = int(lm.y * height)
        cv2.circle(frame, (cx, cy), 4, (0, 80, 255), -1, cv2.LINE_AA)


def draw_overlay(
    frame,
    fps: float,
    inference_ms: float,
    cpu_pct: float,
    ram_mb: float,
) -> None:
    """
    Escribe las métricas y el nombre del modelo en la esquina superior
    izquierda del frame. Operación in-place.
    """
    lines = [
        ("Modelo: MediaPipe Pose Lite",      (10,  28), 0.60, (0, 255, 255)),
        (f"FPS: {fps:.1f}",                  (10,  58), 0.60, (0, 255,   0)),
        (f"Latencia: {inference_ms:.1f} ms", (10,  88), 0.60, (0, 255,   0)),
        (f"CPU: {cpu_pct:.1f}%",             (10, 116), 0.55, (255, 200,  0)),
        (f"RAM: {ram_mb:.0f} MB",            (10, 142), 0.55, (255, 200,  0)),
    ]
    for text, org, scale, color in lines:
        cv2.putText(
            frame, text, org,
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Bucle principal
# ──────────────────────────────────────────────────────────────────────────────

def run(source_arg: str) -> None:
    is_webcam, source = parse_source(source_arg)

    # ── Abrir fuente de video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Error: no se pudo abrir la fuente '{source_arg}'.")

    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    # ── Verificar modelo ───────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        cap.release()
        sys.exit(f"Error: modelo no encontrado en '{MODEL_PATH}'.")

    # ── Crear directorio de resultados ─────────────────────────────────────────
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # ── Inicializar Pose Landmarker ────────────────────────────────────────────
    # Se usa VIDEO mode para ambas fuentes porque:
    #   · Es síncrono → latencia medible directamente con time.perf_counter().
    #   · Permite tracking entre frames (mejor que IMAGE mode).
    #   · No requiere callbacks (más simple que LIVE_STREAM).
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,   # reduce carga de cómputo
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ── Cabecera en terminal ───────────────────────────────────────────────────
    source_label = str(source) if is_webcam else source_arg
    print(
        f"\n{'=' * 40}\n"
        f"Benchmark de estimación de pose\n"
        f"Modelo: MediaPipe Pose Landmarker Lite\n"
        f"Dispositivo: CPU\n"
        f"Fuente: {source_label}\n"
        f"Resolución: {width}x{height}\n"
        f"{'=' * 40}\n"
        f"Controles: [q] o [ESC] para salir\n"
    )

    # ── Estado del proceso ─────────────────────────────────────────────────────
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)   # primera llamada inicializa el contador interno

    fps_counter     = FpsCounter(window=30)
    frame_count     = 0               # frames leídos y procesados correctamente
    detected_frames = 0
    last_ts_ms      = -1              # garantiza timestamps estrictamente crecientes
    start_mono      = time.monotonic()

    # Listas de escalares para el resumen final (NO almacenamos frames)
    fps_stats:     list = []
    latency_stats: list = []
    cpu_stats:     list = []
    ram_stats:     list = []

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "timestamp", "frame", "fps", "inference_ms",
            "cpu_percent", "ram_mb", "pose_detected", "width", "height",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("Error al leer frame de la cámara.", file=sys.stderr)
                else:
                    print("\nFin del video.")
                break

            # Incrementar aquí para que total_frames sea siempre correcto,
            # independientemente de cómo salga el bucle (fin de video o tecla q)
            frame_count += 1
            frame_idx    = frame_count - 1   # índice base-0 para timestamps y CSV

            # ── Timestamp para MediaPipe VIDEO mode ───────────────────────────
            # detect_for_video() exige enteros en milisegundos estrictamente
            # crecientes. Calculamos de forma diferente según la fuente:
            #   · Webcam: tiempo real transcurrido (monotonic) → siempre creciente.
            #   · Video:  posición sintética basada en índice × duración de frame
            #             → monotónica y precisa aunque el codec reporte POS_MSEC=0.
            if is_webcam:
                ts_ms = int((time.monotonic() - start_mono) * 1000)
            else:
                ts_ms = int(frame_idx * 1000.0 / source_fps)

            # Garantía adicional de monotonía (protege contra resolución de reloj)
            ts_ms = max(ts_ms, last_ts_ms + 1)
            last_ts_ms = ts_ms

            # ── Conversión de color y empaquetado de imagen ───────────────────
            # MediaPipe Tasks requiere imagen en formato RGB.
            # cv2.cvtColor crea una copia; no hay forma de evitarlo.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # ── Inferencia ────────────────────────────────────────────────────
            # SÓLO se mide el tiempo del detector; quedan excluidos:
            # cv2.imshow, waitKey, putText y escritura en CSV.
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, ts_ms)
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0

            # ── Detección y dibujo de pose ────────────────────────────────────
            pose_detected = 0
            if result.pose_landmarks:
                pose_detected = 1
                detected_frames += 1
                draw_pose(frame, result.pose_landmarks[0], height, width)

            # ── FPS ───────────────────────────────────────────────────────────
            fps = fps_counter.tick()

            # ── Métricas del proceso ──────────────────────────────────────────
            cpu_pct = proc.cpu_percent(interval=None)
            ram_mb  = proc.memory_info().rss / (1024.0 * 1024.0)

            # ── Registro CSV ──────────────────────────────────────────────────
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            writer.writerow({
                "timestamp":     ts_str,
                "frame":         frame_idx,
                "fps":           round(fps, 2),
                "inference_ms":  round(inference_ms, 2),
                "cpu_percent":   round(cpu_pct, 1),
                "ram_mb":        round(ram_mb, 1),
                "pose_detected": pose_detected,
                "width":         width,
                "height":        height,
            })

            # Flush periódico para robustez ante cortes de energía o Ctrl+C
            if frame_count % CSV_FLUSH_INTERVAL == 0:
                csv_file.flush()

            # ── Acumular estadísticas para el resumen ─────────────────────────
            if fps > 0.0:
                fps_stats.append(fps)
            latency_stats.append(inference_ms)
            cpu_stats.append(cpu_pct)
            ram_stats.append(ram_mb)

            # ── Overlay y visualización ───────────────────────────────────────
            draw_overlay(frame, fps, inference_ms, cpu_pct, ram_mb)
            cv2.imshow(WINDOW_TITLE, frame)

            # waitKey(1): mínimo esperable, sin limitar el procesamiento
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # 27 = ESC
                print("\nDetenido por el usuario.")
                break

    # ── Limpieza ───────────────────────────────────────────────────────────────
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    elapsed  = time.monotonic() - start_mono
    det_pct  = (detected_frames / frame_count * 100.0) if frame_count > 0 else 0.0

    # ── Resumen en terminal ────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print("RESUMEN")
    print(f"{'=' * 40}")
    print(f"Modelo:              MediaPipe Pose Landmarker Lite")
    print(f"Fuente:              {source_label}")
    print(f"Resolución:          {width}x{height}")
    print(f"Frames totales:      {frame_count}")
    print(f"Frames con pose:     {detected_frames}  ({det_pct:.1f}%)")

    if fps_stats:
        avg_fps = sum(fps_stats) / len(fps_stats)
        print(f"FPS promedio:        {avg_fps:.2f}")
        print(f"FPS mínimo:          {min(fps_stats):.2f}")
        print(f"FPS máximo:          {max(fps_stats):.2f}")

    if latency_stats:
        avg_lat = sum(latency_stats) / len(latency_stats)
        print(f"Latencia promedio:   {avg_lat:.2f} ms")
        print(f"Latencia mínima:     {min(latency_stats):.2f} ms")
        print(f"Latencia máxima:     {max(latency_stats):.2f} ms")
        print(f"Latencia P95:        {percentile(latency_stats, 95):.2f} ms")

    if cpu_stats:
        print(f"CPU promedio:        {sum(cpu_stats) / len(cpu_stats):.1f}%")
        print(f"CPU máximo:          {max(cpu_stats):.1f}%")

    if ram_stats:
        print(f"RAM promedio:        {sum(ram_stats) / len(ram_stats):.1f} MB")
        print(f"RAM máximo:          {max(ram_stats):.1f} MB")

    print(f"Tiempo total:        {elapsed:.1f} s")
    print(f"{'=' * 40}")
    print(f"Resultados guardados en: {CSV_PATH}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Uso:\n"
            "  python benchmark/mediapipe_pose.py 0                    # webcam\n"
            "  python benchmark/mediapipe_pose.py 2                    # webcam USB\n"
            "  python benchmark/mediapipe_pose.py video/video1.mp4     # video\n",
            file=sys.stderr,
        )
        sys.exit(1)

    run(sys.argv[1])
