"""Runs the semantic segmentation model and draws its mask.

The model comes from the Edge Impulse custom learning block
(model_quantized_int8_io.tflite or model.tflite). Its output is
1 x H/8 x W/8 x classes softmax: channel 0 = the background (the labeler's
default class), then one channel per Studio label in Studio's order. The
tflite file doesn't carry the names, so they come from the block's
metrics.json (or a labels.txt) next to the model. The Edge Impulse runner
would merge the cells into boxes, so the tensor is read directly with
LiteRT here. Used by the App Lab app (main.py) and the upload app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # some ai-edge-litert wheels don't load on macOS; TensorFlow has the same interpreter
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        raise ImportError("Install ai-edge-litert (or tensorflow) to run the model") from None

# RGB for the shown classes in order (the ignore channel is skipped); the
# background is left clear. Same palette as the learning block's scripts/visualize.py.
PALETTE = [(230, 40, 60), (250, 200, 30), (40, 120, 230), (60, 200, 90), (200, 60, 220),
           (250, 130, 30), (30, 210, 210), (150, 90, 40), (250, 120, 180), (120, 120, 250)]
ALPHA = 0.5
NEAR_ROWS = 0.3  # bottom share of the grid used for the free-space readout


@dataclass
class Classes:
    """Output channel names. ``ignore``: the channel the block never trains, if any."""
    names: list[str]
    ignore: int | None = None
    source: str = "default"

    @property
    def foreground(self) -> list[int]:
        """Channels that can be predicted, except the background."""
        return [i for i in range(1, len(self.names)) if i != self.ignore]

    def colour(self, channel: int) -> tuple[int, int, int]:
        return PALETTE[self.foreground.index(channel) % len(PALETTE)]

    def legend(self) -> list[dict]:
        return [{"channel": i, "name": self.names[i], "colour": "#%02x%02x%02x" % self.colour(i)}
                for i in self.foreground]


def load_classes(models_dir: Path, channels: int) -> Classes:
    """Channel names from metrics.json (the learning block's report) or labels.txt
    (one name per line, channel 0 first), else placeholders."""
    metrics, labels = models_dir / "metrics.json", models_dir / "labels.txt"
    names, ignore_name, source = None, "ignore", "default"
    if metrics.is_file():
        report = json.loads(metrics.read_text())
        names, ignore_name, source = report.get("channels"), report.get("ignore"), metrics.name
    elif labels.is_file():
        names = [line.strip() for line in labels.read_text().splitlines() if line.strip()]
        source = labels.name
    if names is not None and len(names) != channels:
        print(f"{source} lists {len(names)} classes but the model has {channels} outputs; ignoring it", flush=True)
        names = None
    if names is None:
        names, source = ["background", *[f"class {i}" for i in range(1, channels)]], "default"
    ignore = names.index(ignore_name) if ignore_name in names[1:] else None
    return Classes(list(names), ignore, source)


class Segmenter:
    def __init__(self, model_path: str | Path, threads: int = 4):
        self.interp = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.height, self.width, self.channels = self.inp["shape"]
        _, self.rows, self.cols, self.classes = self.out["shape"]

    def probabilities(self, frame_bgr: np.ndarray) -> np.ndarray:
        """(rows, cols, classes) float probabilities for one camera frame.

        The frame is squashed to the model input, like the impulse's Squash resize.
        """
        img = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0

        if self.inp["dtype"] == np.int8:
            scale, zero = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()

        y = self.interp.get_tensor(self.out["index"])[0]
        if self.out["dtype"] == np.int8:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y.astype(np.float32)


def class_map(probs: np.ndarray, classes: Classes, threshold: float) -> np.ndarray:
    """(rows, cols) channel per cell: the most likely class but the background
    wherever those classes together reach the threshold, else 0 (background)."""
    fg = classes.foreground
    if not fg:
        return np.zeros(probs.shape[:2], np.int32)
    p = probs[..., fg]
    return np.where(p.sum(-1) >= threshold, np.asarray(fg)[p.argmax(-1)], 0).astype(np.int32)


def coverage(cmap: np.ndarray, classes: Classes) -> dict[str, float]:
    """Share of cells per class, background included."""
    return {classes.names[i]: float(np.mean(cmap == i)) for i in [0, *classes.foreground]}


def free_space(cmap: np.ndarray) -> list[float]:
    """Share of background in the left, centre and right thirds of the rows nearest the camera."""
    near = cmap[-max(1, round(cmap.shape[0] * NEAR_ROWS)):] == 0
    return [float(part.mean()) for part in np.array_split(near, 3, axis=1)]


def draw_mask(frame_bgr: np.ndarray, cmap: np.ndarray, classes: Classes, near_line: bool = True) -> np.ndarray:
    """The frame with each class's cells in its colour and the output grid on top."""
    h, w = frame_bgr.shape[:2]
    up = cv2.resize(cmap.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    out = frame_bgr.astype(np.float32)
    for ch in classes.foreground:
        m = up == ch
        if m.any():
            r, g, b = classes.colour(ch)
            out[m] = (1 - ALPHA) * out[m] + ALPHA * np.array((b, g, r), np.float32)

    rows, cols = cmap.shape
    for r in range(1, rows):
        y = r * h // rows
        out[y] = out[y] * 0.75 + 255 * 0.25
    for c in range(1, cols):
        x = c * w // cols
        out[:, x] = out[:, x] * 0.75 + 255 * 0.25

    if near_line:  # rows used for the free-space readout
        y = h - round(rows * NEAR_ROWS) * h // rows
        cv2.line(out, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    return out.astype(np.uint8)
