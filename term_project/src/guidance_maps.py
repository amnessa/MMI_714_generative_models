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

    # Robust depth preprocessing
    depth_max_distance: float = 10.0      # Far plane clamp (meters)
    depth_use_inverse: bool = True        # Use 1/z (disparity) for better near-object edges
    depth_use_canny: bool = True          # Use Canny instead of Sobel for depth
    depth_canny_thresh1: int = 30         # Lower threshold for depth Canny
    depth_canny_thresh2: int = 100        # Upper threshold for depth Canny
    depth_edge_dilate: int = 0            # Dilation iterations (0=none, 1-2 for more coverage)

    # SAM configuration
    # NOTE: SAM cannot run in DataLoader workers (CUDA context issues).
    # Enable SAM only for single-threaded/inference use or precomputation.
    use_sam: bool = True                  # Master switch (if False, disables all SAM)
    use_sam_rgb: bool = True              # Use SAM for RGB edges (high quality)
    use_sam_depth: bool = True            # Use SAM for depth edges (high quality)
    sam_model_name: str = "facebook/sam-vit-base"
    sam_points_per_side: int = 32         # Grid density for automatic mask generation
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    sam_device: str = "cuda"              # or "cpu"


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
    Handles inverted masks by checking which representation has more boundary pixels.

    Args:
        masks: (N, H, W) binary masks

    Returns:
        (H, W) boundary map where edges between segments are marked (always white=edge)
    """
    if masks is None or len(masks) == 0:
        return None

    h, w = masks.shape[1], masks.shape[2]
    boundary_map = np.zeros((h, w), dtype=np.float32)

    for mask in masks:
        # Normalize mask to binary (handle both 0-1 and 0-255 ranges)
        if mask.max() > 1:
            mask_norm = mask / 255.0
        else:
            mask_norm = mask

        # Handle potential inversion: check which interpretation gives boundaries
        # (not filled regions)
        mask_uint8 = (mask_norm > 0.5).astype(np.uint8) * 255

        # Use morphological gradient to find boundaries
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel)

        # If the gradient is mostly filled (>30% of image), the mask might be inverted
        # In that case, we should use the inverse
        fill_ratio = gradient.sum() / (h * w * 255)
        if fill_ratio > 0.3:
            # Likely inverted - use inverse mask
            mask_uint8_inv = ((1.0 - mask_norm) > 0.5).astype(np.uint8) * 255
            gradient = cv2.morphologyEx(mask_uint8_inv, cv2.MORPH_GRADIENT, kernel)

        boundary_map = np.maximum(boundary_map, gradient.astype(np.float32) / 255.0)

    return boundary_map


def sam_edges_rgb(rgb: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    model, processor = _get_sam_model(cfg)

    # Ensure RGB format and uint8
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 255)).astype(np.uint8)

    if rgb.shape[2] == 3:
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb_input = rgb

    h, w = rgb.shape[:2]

    # --- OPTIMIZATION START: Compute Image Embeddings ONCE ---
    # We process the image alone first to get the embeddings
    inputs_image = processor(rgb_input, return_tensors="pt").to(cfg.sam_device)
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(inputs_image["pixel_values"])
    # --- OPTIMIZATION END ---

    # Generate grid of input points
    points_per_side = cfg.sam_points_per_side
    x_coords = np.linspace(0, w - 1, points_per_side, dtype=np.int32)
    y_coords = np.linspace(0, h - 1, points_per_side, dtype=np.int32)

    input_points = []
    for y in y_coords:
        for x in x_coords:
            input_points.append([[int(x), int(y)]])

    all_masks = []
    batch_size = 16  # You can likely increase this to 32 or 64 now

    with torch.no_grad():
        for i in range(0, len(input_points), batch_size):
            batch_points = input_points[i:i + batch_size]

            # Use processor to format points, but we will DISCARD the pixel_values
            # so the model uses our cached embeddings instead.
            inputs = processor(
                rgb_input,
                input_points=[batch_points], # Note: processor handles list of lists
                return_tensors="pt"
            )

            # Remove pixel_values so model doesn't re-run encoder
            inputs.pop("pixel_values", None)

            # Add our cached embeddings
            inputs["image_embeddings"] = image_embeddings

            # Move remaining inputs (like input_points) to device
            inputs = {k: v.to(cfg.sam_device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Get masks
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
        return rgb_edges(rgb, cfg)

    all_masks = np.array(all_masks)
    boundary_map = _extract_boundaries_from_masks(all_masks)
    boundary_map = _morph_close((boundary_map * 255).astype(np.uint8), cfg.morph_kernel)
    boundary_map = (boundary_map > 0).astype(np.float32)

    return boundary_map


def sam_edges_depth(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Use SAM to generate boundary/edge map from depth image.
    Uses robust preprocessing: sanitization + inverse depth for better edges.
    """
    model, processor = _get_sam_model(cfg)

    # Robust depth preprocessing (same as depth_edges)
    d = depth.astype(np.float32)

    # Sanitize NaN/Inf
    d = np.nan_to_num(d, nan=cfg.depth_max_distance,
                       posinf=cfg.depth_max_distance,
                       neginf=0.0)
    d = np.clip(d, 0, cfg.depth_max_distance)

    # Convert to inverse depth for better near-object representation
    if cfg.depth_use_inverse:
        epsilon = 1e-3
        d_inv = 1.0 / (d + epsilon)
        d_min, d_max = d_inv.min(), d_inv.max()
        if d_max - d_min > 1e-6:
            d_norm = (d_inv - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d_inv)
    else:
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)

    depth_uint8 = (d_norm * 255).astype(np.uint8)
    depth_rgb = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)

    h, w = depth.shape[:2]

    # --- OPTIMIZATION START ---
    inputs_image = processor(depth_rgb, return_tensors="pt").to(cfg.sam_device)
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(inputs_image["pixel_values"])
    # --- OPTIMIZATION END ---

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

            inputs = processor(
                depth_rgb,
                input_points=[batch_points],
                return_tensors="pt"
            )

            inputs.pop("pixel_values", None)
            inputs["image_embeddings"] = image_embeddings
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
    """
    Robust edge detection on depth maps.

    Handles the challenges of raw EXR depth data:
    1. Sanitizes NaN/Inf values (sensor errors, sky)
    2. Optionally converts to inverse depth (1/z) for better near-object edges
    3. Normalizes to 0-255 range for edge detection
    4. Uses Canny (more robust) or Sobel for edge extraction

    Args:
        depth: (H, W) raw depth map (can contain NaN, Inf, large values)
        cfg: GuidanceConfig with depth preprocessing parameters

    Returns:
        (H, W) binary edge map as float32 in [0, 1]
    """
    d = depth.astype(np.float32)

    # Step 1: Create validity mask (before sanitization)
    valid_mask = np.isfinite(d) & (d > 0)

    # Step 2: Sanitize - replace NaN/Inf with far plane, clip to max distance
    d = np.nan_to_num(d, nan=cfg.depth_max_distance,
                       posinf=cfg.depth_max_distance,
                       neginf=0.0)
    d = np.clip(d, 0, cfg.depth_max_distance)

    # Step 3: Convert to inverse depth (disparity) if enabled
    # Inverse depth emphasizes near objects and compresses far objects
    # This is crucial because transparent objects are usually close to the camera
    if cfg.depth_use_inverse:
        # Add epsilon to avoid division by zero
        epsilon = 1e-3
        d_inv = 1.0 / (d + epsilon)
        # Normalize inverse depth to [0, 1]
        d_min, d_max = d_inv.min(), d_inv.max()
        if d_max - d_min > 1e-6:
            d_norm = (d_inv - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d_inv)
    else:
        # Linear normalization
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)

    # Step 4: Convert to uint8 for edge detection
    d_uint8 = (d_norm * 255).astype(np.uint8)

    # Step 5: Apply Gaussian blur to reduce noise
    d_uint8 = cv2.GaussianBlur(d_uint8, (3, 3), 0)

    # Step 6: Edge detection
    if cfg.depth_use_canny:
        # Canny is more robust and produces cleaner edges
        edges = cv2.Canny(d_uint8, cfg.depth_canny_thresh1, cfg.depth_canny_thresh2,
                          apertureSize=cfg.rgb_canny_aperture)
    else:
        # Fallback to Sobel
        gx = cv2.Sobel(d_uint8, cv2.CV_32F, 1, 0, ksize=cfg.depth_sobel_ksize)
        gy = cv2.Sobel(d_uint8, cv2.CV_32F, 0, 1, ksize=cfg.depth_sobel_ksize)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = np.percentile(mag, cfg.depth_edge_percentile)
        edges = (mag >= thr).astype(np.uint8) * 255

    # Step 7: Morphological closing to connect nearby edges
    edges = _morph_close(edges, cfg.morph_kernel)

    # Step 8: Optional dilation to increase overlap with RGB edges
    if cfg.depth_edge_dilate > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=cfg.depth_edge_dilate)

    return (edges > 0).astype(np.float32)


# ============================================================================
# Main guidance map functions
# ============================================================================

def m_da(rgb_bgr: np.ndarray, depth: np.ndarray, cfg: GuidanceConfig,
         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Depth-Aware boundary map for OPTICAL inpainting (Track A).

    Per DITR paper: M_DA = M_RGB \\\\ C_U(M_D) = M_RGB ∩ M_D (INTERSECTION)

    Mathematical proof:
        M_DA = M_RGB \\\\ C_U(M_D)  where C_U is complement
        A \\\\ B^c = A ∩ B

    This finds edges that appear in BOTH RGB and Depth ("confirmed" real boundaries).
    Edges only in RGB (not in M_DA) are likely transparent/reflective ghost edges.

    Args:
        rgb_bgr: (H, W, 3) BGR image
        depth: (H, W) depth map
        cfg: GuidanceConfig
        mask: Optional (H, W) mask where 1=transparent area

    Returns:
        (H, W) guidance map - intersection of RGB and Depth edges, masked to transparent region
    """
    # RGB edges: use SAM if enabled for RGB
    if cfg.use_sam and cfg.use_sam_rgb:
        e_rgb = sam_edges_rgb(rgb_bgr, cfg)
    else:
        e_rgb = rgb_edges(rgb_bgr, cfg)

    # Depth edges: use SAM if enabled for depth
    if cfg.use_sam and cfg.use_sam_depth:
        e_d = sam_edges_depth(depth, cfg)
    else:
        e_d = depth_edges(depth, cfg)

    # M_DA = M_RGB ∩ M_D (INTERSECTION, not difference!)
    # This gives us edges that are confirmed in both modalities
    out = np.minimum(e_rgb, e_d)  # Intersection for binary maps

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
    # RGB edges: use SAM if enabled for RGB
    if cfg.use_sam and cfg.use_sam_rgb:
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
            if self.cfg.use_sam and self.cfg.use_sam_rgb:
                self._rgb_cache[key] = sam_edges_rgb(rgb_bgr, self.cfg)
            else:
                self._rgb_cache[key] = rgb_edges(rgb_bgr, self.cfg)

        return self._rgb_cache[key]

    def get_depth_edges(self, depth: np.ndarray, key: Union[str, int, None] = None) -> np.ndarray:
        """Get depth edges, using cache if available."""
        if key is None:
            key = id(depth)

        if key not in self._depth_cache:
            if self.cfg.use_sam and self.cfg.use_sam_depth:
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
