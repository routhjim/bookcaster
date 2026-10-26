"""Sky mask for an equirect panorama with the skyseg ONNX model (HY-World's WorldMirror ships it), CPU.

MoGe-2's own mask misses dark night skies: they come back as geometry at a few metres, a flat "lid" over the
camera (10-20, two night exteriors). The model runs on overlapping 90-degree windows of the upper half
(little distortion there), the scores are averaged, and only pixels ABOVE the horizon can become sky.

    sky = sky_mask_equirect(rgb_uint8)            # (H, W) bool
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SKYSEG = Path(os.environ.get("WORLD_SKYSEG", Path.home() / "models" / "hy-world" / "skyseg.onnx"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _session():
    import onnxruntime as ort
    return ort.InferenceSession(str(SKYSEG), providers=["CPUExecutionProvider"])


def _score(sess, rgb):
    """Model output for one RGB window, normalised to 0..1 (HIGH = sky: HY-World's visual_util marks < 32/255
    as NON-sky, despite its comment), at the window's size."""
    import cv2
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (320, 320)).astype(np.float32) / 255
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = np.asarray(sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})[0]).squeeze()
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def sky_mask_equirect(rgb: np.ndarray, threshold: float = 32 / 255, windows: int = 8) -> np.ndarray:
    """(H, W) bool sky mask of an equirect RGB uint8 panorama (column 0 = longitude -180)."""
    H, W = rgb.shape[:2]
    top = rgb[: H // 2]                                   # latitude +90 .. 0 (the upper hemisphere)
    win = W // 4                                          # 90 degrees wide
    acc = np.zeros(top.shape[:2], np.float32)
    cnt = np.zeros(top.shape[:2], np.float32)
    sess = _session()
    for k in range(windows):
        c0 = int(k * W / windows)
        cols = (np.arange(c0, c0 + win)) % W
        s = _score(sess, np.ascontiguousarray(top[:, cols]))
        np.add.at(acc, (slice(None), cols), s)
        np.add.at(cnt, (slice(None), cols), 1)
    score = acc / np.maximum(cnt, 1)
    sky = np.zeros((H, W), bool)
    sky[: H // 2] = score >= threshold
    return sky
