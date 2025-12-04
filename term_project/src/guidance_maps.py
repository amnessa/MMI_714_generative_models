import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, List, Union
import torch


@dataclass
class GuidanceConfig:
    # Classic edge detection params (fallback)
    rgb_canny_thresh1: int = 100
    rgb_canny_thresh2: int = 200
    rgb_canny_aperture: int = 3
    depth_sobel_ksize: int = 3
    depth_edge_percentile: float = 90.0
    morph_kernel: int = 3

    # SAM configuration
    use_sam: bool = False
    sam_model_name: str = "facebook/sam-vit-large"
    sam_points_per_side: int = 32  # Grid density for automatic mask generation
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    sam_device: str = "cuda"  # or "cpu"


# Global SAM model cache to avoid reloading
_SAM_MODEL_CACHE = {}


def _get_sam_model(cfg: GuidanceConfig):
    """Lazy load and cache SAM model."""
    global _SAM_MODEL_CACHE

    cache_key = (cfg.sam_model_name, cfg.sam_device)
    if cache_key not in _SAM_MODEL_CACHE:
        try:
            from transformers import SamModel, SamProcessor

            print(f"Loading SAM model: {cfg.sam_model_name}...")
            processor = SamProcessor.from_pretrained(cfg.sam_model_name)
            model = SamModel.from_pretrained(cfg.sam_model_name)
            model = model.to(cfg.sam_device)
            model.eval()

            _SAM_MODEL_CACHE[cache_key] = (model, processor)
            print("SAM model loaded successfully.")
        except ImportError:
            raise ImportError(
                "SAM requires 'transformers' package. Install with: "
                "pip install transformers"
            )

    return _SAM_MODEL_CACHE[cache_key]


def _morph_close(bin_img: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return bin_img
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)


def _extract_boundaries_from_masks(masks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract boundary/edge map from a set of segmentation masks.

    Args:
        masks: (N, H, W) binary masks

    Returns:
        (H, W) boundary map where edges between segments are marked
    """
    if masks is None or len(masks) == 0:
        return None

    h, w = masks.shape[1], masks.shape[2]
    boundary_map = np.zeros((h, w), dtype=np.float32)

    for mask in masks:
        # Find contours/edges of each mask
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        # Use morphological gradient to find boundaries
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel)
        boundary_map = np.maximum(boundary_map, gradient.astype(np.float32) / 255.0)

    return boundary_map


def sam_edges_rgb(rgb: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Use SAM to generate boundary/edge map from RGB image.

    Args:
        rgb: (H, W, 3) RGB image (uint8, 0-255)
        cfg: GuidanceConfig with SAM parameters

    Returns:
        (H, W) boundary map as float32 in [0, 1]
    """
    model, processor = _get_sam_model(cfg)

    # Ensure RGB format and uint8
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 255)).astype(np.uint8)

    # SAM expects RGB, if BGR convert
    if rgb.shape[2] == 3:
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if cfg else rgb
    else:
        rgb_input = rgb

    h, w = rgb.shape[:2]

    # Generate grid of input points for automatic segmentation
    points_per_side = cfg.sam_points_per_side
    x_coords = np.linspace(0, w - 1, points_per_side, dtype=np.int32)
    y_coords = np.linspace(0, h - 1, points_per_side, dtype=np.int32)

    # Create grid of points
    input_points = []
    for y in y_coords:
        for x in x_coords:
            input_points.append([[int(x), int(y)]])

    # Process in batches to avoid OOM
    all_masks = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(input_points), batch_size):
            batch_points = input_points[i:i + batch_size]

            # Process each point
            for points in batch_points:
                inputs = processor(
                    rgb_input,
                    input_points=[points],
                    return_tensors="pt"
                )
                inputs = {k: v.to(cfg.sam_device) for k, v in inputs.items()}

                outputs = model(**inputs)

                # Get masks
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]

                # Filter by IoU score
                iou_scores = outputs.iou_scores.cpu().numpy()[0]
                for j, score in enumerate(iou_scores[0]):
                    if score > cfg.sam_pred_iou_thresh:
                        mask = masks[0, j].numpy()
                        all_masks.append(mask)

    if len(all_masks) == 0:
        # Fallback to classical edge detection
        return rgb_edges(rgb, cfg)

    all_masks = np.array(all_masks)
    boundary_map = _extract_boundaries_from_masks(all_masks)

    # Apply morphological closing
    boundary_map = _morph_close((boundary_map * 255).astype(np.uint8), cfg.morph_kernel)
    boundary_map = (boundary_map > 0).astype(np.float32)

    return boundary_map


def sam_edges_depth(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Use SAM to generate boundary/edge map from depth image.
    Depth is converted to a pseudo-RGB representation for SAM.

    Args:
        depth: (H, W) depth image as float32
        cfg: GuidanceConfig with SAM parameters

    Returns:
        (H, W) boundary map as float32 in [0, 1]
    """
    model, processor = _get_sam_model(cfg)

    # Normalize depth to 0-255 range
    d = depth.astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d_norm = (d - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(d)

    # Convert to pseudo-RGB (grayscale replicated 3 times)
    depth_uint8 = (d_norm * 255).astype(np.uint8)
    depth_rgb = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)

    h, w = depth.shape[:2]

    # Generate grid of input points
    points_per_side = cfg.sam_points_per_side
    x_coords = np.linspace(0, w - 1, points_per_side, dtype=np.int32)
    y_coords = np.linspace(0, h - 1, points_per_side, dtype=np.int32)

    input_points = []
    for y in y_coords:
        for x in x_coords:
            input_points.append([[int(x), int(y)]])

    all_masks = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(input_points), batch_size):
            batch_points = input_points[i:i + batch_size]

            for points in batch_points:
                inputs = processor(
                    depth_rgb,
                    input_points=[points],
                    return_tensors="pt"
                )
                inputs = {k: v.to(cfg.sam_device) for k, v in inputs.items()}

                outputs = model(**inputs)

                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]

                iou_scores = outputs.iou_scores.cpu().numpy()[0]
                for j, score in enumerate(iou_scores[0]):
                    if score > cfg.sam_pred_iou_thresh:
                        mask = masks[0, j].numpy()
                        all_masks.append(mask)

    if len(all_masks) == 0:
        # Fallback to classical edge detection
        return depth_edges(depth, cfg)

    all_masks = np.array(all_masks)
    boundary_map = _extract_boundaries_from_masks(all_masks)

    boundary_map = _morph_close((boundary_map * 255).astype(np.uint8), cfg.morph_kernel)
    boundary_map = (boundary_map > 0).astype(np.float32)

    return boundary_map


# ============================================================================
# Classic edge detection (fallback when SAM is disabled)
# ============================================================================

def rgb_edges(rgb_bgr: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """Classical Canny edge detection on RGB."""
    if rgb_bgr.ndim == 3 and rgb_bgr.shape[2] == 3:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_bgr
    edges = cv2.Canny(gray, cfg.rgb_canny_thresh1, cfg.rgb_canny_thresh2,
                      apertureSize=cfg.rgb_canny_aperture)
    edges = _morph_close(edges, cfg.morph_kernel)
    return (edges > 0).astype(np.float32)


def depth_edges(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """Classical Sobel edge detection on depth."""
    d = depth.astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=cfg.depth_sobel_ksize)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=cfg.depth_sobel_ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, cfg.depth_edge_percentile)
    edges = (mag >= thr).astype(np.uint8) * 255
    edges = _morph_close(edges, cfg.morph_kernel)
    return (edges > 0).astype(np.float32)


# ============================================================================
# Main guidance map functions
# ============================================================================

def m_da(rgb_bgr: np.ndarray, depth: np.ndarray, cfg: GuidanceConfig,
         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Depth-Aware boundary map for OPTICAL inpainting (Track A).

    Per DITR paper: M_DA = M_RGB \\ C_U(M_D) (set difference)
    This highlights edges visible in RGB but NOT in depth (ghost edges from transparency).

    Args:
        rgb_bgr: (H, W, 3) BGR image
        depth: (H, W) depth map
        cfg: GuidanceConfig
        mask: Optional (H, W) mask where 1=transparent area

    Returns:
        (H, W) guidance map, optionally masked to transparent region only
    """
    if cfg.use_sam:
        e_rgb = sam_edges_rgb(rgb_bgr, cfg)
        e_d = sam_edges_depth(depth, cfg)
    else:
        e_rgb = rgb_edges(rgb_bgr, cfg)
        e_d = depth_edges(depth, cfg)

    # M_DA = RGB edges - Depth edges (set difference)
    out = np.clip(e_rgb - e_d, 0.0, 1.0)

    # Apply mask: guidance only inside transparent region
    if mask is not None:
        out = out * mask

    return out.astype(np.float32)


def m_rgb(rgb_bgr: np.ndarray, cfg: GuidanceConfig,
          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    RGB boundary map for GEOMETRIC inpainting (Track B).

    Args:
        rgb_bgr: (H, W, 3) BGR image
        cfg: GuidanceConfig
        mask: Optional (H, W) mask where 1=transparent area

    Returns:
        (H, W) guidance map, optionally masked to background region only
    """
    if cfg.use_sam:
        edges = sam_edges_rgb(rgb_bgr, cfg)
    else:
        edges = rgb_edges(rgb_bgr, cfg)

    # Apply inverse mask: guidance only outside transparent region (background)
    if mask is not None:
        out = edges * (1.0 - mask)
    else:
        out = edges

    return out.astype(np.float32)


# ============================================================================
# Utility: Precompute and cache SAM edges for a batch
# ============================================================================

class SAMGuidanceCache:
    """
    Cache for precomputed SAM edges to avoid redundant computation.
    Useful when processing the same image multiple times.
    """
    def __init__(self, cfg: GuidanceConfig):
        self.cfg = cfg
        self._rgb_cache = {}
        self._depth_cache = {}

    def get_rgb_edges(self, rgb_bgr: np.ndarray, key: Union[str, int, None] = None) -> np.ndarray:
        """Get RGB edges, using cache if available."""
        if key is None:
            key = id(rgb_bgr)

        if key not in self._rgb_cache:
            if self.cfg.use_sam:
                self._rgb_cache[key] = sam_edges_rgb(rgb_bgr, self.cfg)
            else:
                self._rgb_cache[key] = rgb_edges(rgb_bgr, self.cfg)

        return self._rgb_cache[key]

    def get_depth_edges(self, depth: np.ndarray, key: Union[str, int, None] = None) -> np.ndarray:
        """Get depth edges, using cache if available."""
        if key is None:
            key = id(depth)

        if key not in self._depth_cache:
            if self.cfg.use_sam:
                self._depth_cache[key] = sam_edges_depth(depth, self.cfg)
            else:
                self._depth_cache[key] = depth_edges(depth, self.cfg)

        return self._depth_cache[key]

    def clear(self):
        """Clear the cache."""
        self._rgb_cache.clear()
        self._depth_cache.clear()


if __name__ == "__main__":
    # Quick test
    print("Testing guidance maps...")

    # Create dummy data
    h, w = 128, 128
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32) * 5.0
    mask = np.zeros((h, w), dtype=np.float32)
    mask[40:80, 40:80] = 1.0  # Glass region in center

    # Test with classical edges
    cfg_classic = GuidanceConfig(use_sam=False)
    m_da_classic = m_da(rgb, depth, cfg_classic, mask)
    m_rgb_classic = m_rgb(rgb, cfg_classic, mask)

    print(f"M_DA (classic) shape: {m_da_classic.shape}, range: [{m_da_classic.min():.3f}, {m_da_classic.max():.3f}]")
    print(f"M_RGB (classic) shape: {m_rgb_classic.shape}, range: [{m_rgb_classic.min():.3f}, {m_rgb_classic.max():.3f}]")

    # Test with SAM (if available)
    try:
        cfg_sam = GuidanceConfig(use_sam=True, sam_device="cuda")
        m_da_sam = m_da(rgb, depth, cfg_sam, mask)
        m_rgb_sam = m_rgb(rgb, cfg_sam, mask)
        print(f"M_DA (SAM) shape: {m_da_sam.shape}, range: [{m_da_sam.min():.3f}, {m_da_sam.max():.3f}]")
        print(f"M_RGB (SAM) shape: {m_rgb_sam.shape}, range: [{m_rgb_sam.min():.3f}, {m_rgb_sam.max():.3f}]")
    except Exception as e:
        print(f"SAM test skipped: {e}")
