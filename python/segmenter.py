"""Runs the floor / obstacle segmentation model and draws its mask.

The model comes from the Edge Impulse custom learning block
(model_quantized_int8_io.tflite or model.tflite). Its output is
1 x H/8 x W/8 x classes softmax: channel 0 = floor, channel 1 = obstacle.
The Edge Impulse runner would merge the cells into boxes, so the tensor is
read directly with LiteRT here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

OBSTACLE_BGR = np.array((60, 40, 230), np.float32)  # red
ALPHA = 0.5
NEAR_ROWS = 0.3  # bottom share of the grid used for the free-space readout


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


def obstacle_mask(probs: np.ndarray, threshold: float) -> np.ndarray:
    """(rows, cols) bool: obstacle wherever its probability reaches the threshold."""
    return probs[..., 1] >= threshold


def free_space(mask: np.ndarray) -> list[float]:
    """Share of floor in the left, centre and right thirds of the rows nearest the robot."""
    near = mask[-max(1, round(mask.shape[0] * NEAR_ROWS)):]
    return [float(1 - part.mean()) for part in np.array_split(near, 3, axis=1)]


def draw_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The frame with obstacle cells in red and the output grid on top."""
    h, w = frame_bgr.shape[:2]
    up = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    out = frame_bgr.astype(np.float32)
    out[up] = (1 - ALPHA) * out[up] + ALPHA * OBSTACLE_BGR

    rows, cols = mask.shape
    for r in range(1, rows):
        y = r * h // rows
        out[y] = out[y] * 0.75 + 255 * 0.25
    for c in range(1, cols):
        x = c * w // cols
        out[:, x] = out[:, x] * 0.75 + 255 * 0.25

    # Line marking the rows used for the free-space readout
    y = h - round(rows * NEAR_ROWS) * h // rows
    cv2.line(out, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    return out.astype(np.uint8)
