import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class GuidanceConfig:
    rgb_canny_thresh1: int = 100
    rgb_canny_thresh2: int = 200
    rgb_canny_aperture: int = 3
    depth_sobel_ksize: int = 3
    depth_edge_percentile: float = 90.0  # threshold percentile for gradient magnitude
    morph_kernel: int = 3  # post edge morphology
    use_sam: bool = False  # placeholder; integrate SAM if available


def _morph_close(bin_img: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return bin_img
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)


def rgb_edges(rgb_bgr: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    if rgb_bgr.ndim == 3 and rgb_bgr.shape[2] == 3:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_bgr
    edges = cv2.Canny(gray, cfg.rgb_canny_thresh1, cfg.rgb_canny_thresh2, apertureSize=cfg.rgb_canny_aperture)
    edges = _morph_close(edges, cfg.morph_kernel)
    return (edges > 0).astype(np.float32)


def depth_edges(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    # assume depth is float32, HxW
    d = depth.astype(np.float32)
    # handle inf/nan
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=cfg.depth_sobel_ksize)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=cfg.depth_sobel_ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, cfg.depth_edge_percentile)
    edges = (mag >= thr).astype(np.uint8) * 255
    edges = _morph_close(edges, cfg.morph_kernel)
    return (edges > 0).astype(np.float32)


def m_da(rgb_bgr: np.ndarray, depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    # Depth-Aware boundary map: RGB edges minus Depth edges
    e_rgb = rgb_edges(rgb_bgr, cfg)
    e_d = depth_edges(depth, cfg)
    out = np.clip(e_rgb - e_d, 0.0, 1.0)
    return out.astype(np.float32)


def m_rgb(rgb_bgr: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    return rgb_edges(rgb_bgr, cfg)


# Placeholder for SAM-based edges (optional)
# def sam_edges(rgb_bgr: np.ndarray, depth: np.ndarray, cfg: GuidanceConfig) -> Tuple[np.ndarray, np.ndarray]:
#     raise NotImplementedError("Integrate SAM here if available.")
