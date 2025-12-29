from __future__ import annotations

import cv2
import numpy as np

def _odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

def black_border_mask(frame_bgr: np.ndarray, v_thresh: int = 35, morph_kernel: int = 19) -> np.ndarray:
    """Binary mask (uint8 0/255) of NON-black pixels."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    non_black = (v > v_thresh).astype(np.uint8) * 255

    k = _odd(morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    non_black = cv2.morphologyEx(non_black, cv2.MORPH_CLOSE, kernel, iterations=1)
    non_black = cv2.morphologyEx(non_black, cv2.MORPH_OPEN, kernel, iterations=1)
    return non_black

def green_field_mask(
    frame_bgr: np.ndarray,
    h_min: int = 25,
    h_max: int = 70,
    s_min: int = 80,
    v_min: int = 40,
    morph_kernel: int = 11
) -> np.ndarray:
    """Binary mask (uint8 0/255) of likely field/grass pixels."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= h_min) & (h <= h_max) & (s >= s_min) & (v >= v_min)).astype(np.uint8) * 255

    # gentle ops to avoid connecting non-field green regions to the field
    k = _odd(morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m

def combine_masks(*masks: np.ndarray) -> np.ndarray:
    out = None
    for m in masks:
        if m is None:
            continue
        out = m.copy() if out is None else cv2.bitwise_and(out, m)
    if out is None:
        raise ValueError("No masks provided")
    return out

def apply_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

def keep_component_with_seed(mask: np.ndarray, seed_x: int, seed_y: int) -> np.ndarray:
    """Keep only the foreground connected component containing the seed point.
    mask must be uint8 0/255.
    """
    m = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(m)
    h, w = mask.shape[:2]
    sx = int(np.clip(seed_x, 0, w - 1))
    sy = int(np.clip(seed_y, 0, h - 1))
    lab = int(labels[sy, sx])

    if lab == 0:
        # seed in background; keep largest component
        best_lab = 0
        best_area = 0
        for i in range(1, num):
            area = int((labels == i).sum())
            if area > best_area:
                best_area = area
                best_lab = i
        lab = best_lab

    if lab == 0:
        return mask

    out = (labels == lab).astype(np.uint8) * 255
    return out
