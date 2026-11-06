"""Segmentation viewer: camera on the left, floor / obstacle mask on the right."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App
from arduino.app_utils.image import compress_to_jpeg

from segmenter import Segmenter, draw_mask, free_space, obstacle_mask

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
JPEG_QUALITY = 70


def find_model() -> Path:
    preferred = MODELS_DIR / "model_quantized_int8_io.tflite"
    if preferred.exists():
        return preferred
    models = sorted(MODELS_DIR.glob("*.tflite"))
    if not models:
        raise SystemExit(f"No .tflite model in {MODELS_DIR}. Copy model_quantized_int8_io.tflite there.")
    return models[0]


model_path = find_model()
seg = Segmenter(model_path)
print(f"Model {model_path.name}: input {seg.width}x{seg.height}x{seg.channels}, "
      f"output {seg.cols}x{seg.rows}x{seg.classes}", flush=True)

settings = {"threshold": 0.5}
clients = set()
last_frame_time = None

ui = WebUI()
camera = Camera(resolution=(640, 480), fps=10)


def settings_message():
    return {**settings, "model": model_path.name, "input": f"{seg.width}x{seg.height}",
            "grid": f"{seg.cols}x{seg.rows}"}


def on_connect(sid):
    clients.add(sid)
    ui.send_message("settings", settings_message(), room=sid)


def set_threshold(sid, data):
    try:
        value = float(data.get("threshold"))
    except (AttributeError, TypeError, ValueError):
        return
    settings["threshold"] = min(max(value, 0.05), 0.95)
    ui.send_message("settings", settings_message())


ui.on_connect(on_connect)
ui.on_disconnect(clients.discard)
ui.on_message("set_threshold", set_threshold)


def jpeg_base64(image) -> str | None:
    encoded = compress_to_jpeg(image, quality=JPEG_QUALITY)
    return None if encoded is None else base64.b64encode(encoded.tobytes()).decode("ascii")


def loop():
    global last_frame_time
    frame = camera.capture()
    if frame is None:
        return
    if not clients:
        last_frame_time = None
        time.sleep(0.2)
        return

    try:
        start = time.perf_counter()
        probs = seg.probabilities(frame)
        inference_ms = (time.perf_counter() - start) * 1000

        mask = obstacle_mask(probs, settings["threshold"])
        camera_jpeg = jpeg_base64(frame)
        mask_jpeg = jpeg_base64(draw_mask(frame, mask))
    except Exception as e:
        print(f"Frame failed: {e}", flush=True)
        return
    if camera_jpeg is None or mask_jpeg is None:
        return

    now = time.perf_counter()
    fps = 0.0 if last_frame_time is None else 1 / max(now - last_frame_time, 1e-6)
    last_frame_time = now

    ui.send_message("frame", {
        "camera": camera_jpeg,
        "mask": mask_jpeg,
        "inference_ms": round(inference_ms, 1),
        "fps": round(fps, 1),
        "obstacle": round(float(mask.mean()), 3),
        "free": [round(v, 3) for v in free_space(mask)],
    })


camera.start()
App.run(user_loop=loop)
