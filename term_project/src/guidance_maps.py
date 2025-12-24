import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, List, Union
import torch


@dataclass
class GuidanceConfig:
    # =========================================================================
    # Edge detector selection: 'canny', 'sam', or 'pidinet'
    # =========================================================================
    edge_detector: Literal['canny', 'sam', 'pidinet'] = 'canny'

    # Classic edge detection params (for edge_detector='canny')
    rgb_canny_thresh1: int = 100
    rgb_canny_thresh2: int = 200
    rgb_canny_aperture: int = 3
    depth_sobel_ksize: int = 3
    depth_edge_percentile: float = 90.0
    morph_kernel: int = 3

    # Enhanced edge detection for transparent objects (for edge_detector='canny')
    rgb_use_enhanced: bool = True         # Use multi-channel enhanced edge detection
    rgb_enhance_clahe: bool = True        # Apply CLAHE for local contrast enhancement
    rgb_clahe_clip: float = 2.0           # CLAHE clip limit (higher = more contrast)
    rgb_clahe_grid: int = 8               # CLAHE tile grid size
    rgb_use_lab_channel: bool = True      # Use LAB L-channel (better for glass)
    rgb_use_saturation: bool = False      # Use HSV saturation channel (can be noisy)
    rgb_multi_scale: bool = False         # Combine edges at multiple Canny thresholds
    rgb_canny_low_thresh1: int = 50       # Lower threshold pair for subtle edges
    rgb_canny_low_thresh2: int = 100
    rgb_gradient_magnitude: bool = False  # Also use gradient magnitude thresholding
    rgb_gradient_percentile: float = 90.0 # Percentile for gradient thresholding (higher = fewer edges)

    # Robust depth preprocessing
    depth_max_distance: float = 10.0      # Far plane clamp (meters)
    depth_use_inverse: bool = True        # Use 1/z (disparity) for better near-object edges
    depth_use_canny: bool = True          # Use Canny instead of Sobel for depth
    depth_canny_thresh1: int = 30         # Lower threshold for depth Canny
    depth_canny_thresh2: int = 100        # Upper threshold for depth Canny
    depth_edge_dilate: int = 0            # Dilation iterations (0=none, 1-2 for more coverage)

    # Enhanced edge detection for depth images
    depth_use_enhanced: bool = True       # Use enhanced edge detection for depth
    depth_enhance_clahe: bool = True      # Apply CLAHE for local contrast enhancement
    depth_clahe_clip: float = 2.0         # CLAHE clip limit
    depth_clahe_grid: int = 8             # CLAHE tile grid size
    depth_multi_scale: bool = True        # Combine edges at multiple Canny thresholds
    depth_canny_low_thresh1: int = 15     # Lower threshold pair for subtle depth edges
    depth_canny_low_thresh2: int = 50
    depth_gradient_magnitude: bool = False # Also use gradient magnitude thresholding
    depth_gradient_percentile: float = 85.0 # Percentile for gradient thresholding

    # SAM configuration (for edge_detector='sam')
    # NOTE: SAM cannot run in DataLoader workers (CUDA context issues).
    # Enable SAM only for single-threaded/inference use or precomputation.
    use_sam: bool = False                 # Legacy switch (use edge_detector='sam' instead)
    use_sam_rgb: bool = True              # Use SAM for RGB edges (high quality)
    use_sam_depth: bool = True            # Use SAM for depth edges (high quality)
    sam_model_name: str = "facebook/sam-vit-base"
    sam_points_per_side: int = 48         # Grid density (higher = more coverage, slower)
    sam_pred_iou_thresh: float = 0.70     # Lower to catch transparent objects (was 0.88)
    sam_stability_score_thresh: float = 0.80  # Lower for glass objects (was 0.95)
    sam_device: str = "cuda"              # or "cpu"
    sam_use_all_masks: bool = True        # Use all 3 mask scales (small/medium/large)
    sam_combine_with_canny: bool = True   # Combine SAM boundaries with Canny edges

    # PiDiNet configuration (for edge_detector='pidinet')
    pidinet_weights_path: Optional[str] = None  # Path to pretrained weights (.pth)
    pidinet_threshold: float = 0.5              # Edge probability threshold
    pidinet_device: str = "cuda"
    pidinet_dilation: int = 24                  # CDCM dilation channels (must match pretrained weights)
    pidinet_morph_erode: int = 1                # Erosion iterations to thin edges (0=disabled)


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

    For each mask, we find the contour (boundary) and add it to the boundary map.
    This gives us edges where segments meet, which is what we want for M_RGB and M_D.

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
        # Normalize mask to binary uint8
        if mask.max() > 1:
            mask_uint8 = (mask > 127).astype(np.uint8) * 255
        else:
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        # Find contours - these are the actual boundaries
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw contours as boundaries (1 pixel thick)
        contour_img = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_img, contours, -1, 255, thickness=1)

        # Add to boundary map
        boundary_map = np.maximum(boundary_map, contour_img.astype(np.float32) / 255.0)

    return boundary_map


def sam_edges_rgb(rgb: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Use SAM to generate boundary/edge map from RGB image.

    SAM outputs 3 masks per point (small/medium/large). We collect all of them
    that pass the IoU threshold to maximize boundary coverage.

    For transparent objects like glass, SAM may not segment them well, so we:
    1. Use lower IoU threshold to accept more uncertain masks
    2. Collect all 3 mask scales per point
    3. Optionally combine with Canny edges for complete coverage
    """
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
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(input_points), batch_size):
            batch_points = input_points[i:i + batch_size]

            inputs = processor(
                rgb_input,
                input_points=[batch_points],
                return_tensors="pt"
            )

            inputs.pop("pixel_values", None)
            inputs["image_embeddings"] = image_embeddings
            inputs = {k: v.to(cfg.sam_device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Get masks - shape is (batch, num_points, 3, H, W) where 3 = mask scales
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]  # First batch

            iou_scores = outputs.iou_scores.cpu().numpy()  # Shape: (batch, num_points, 3)

            # Iterate over each point in the batch
            for point_idx in range(masks.shape[0]):
                point_iou_scores = iou_scores[0, point_idx]  # 3 scores for 3 mask scales

                if cfg.sam_use_all_masks:
                    # Collect ALL 3 mask scales (small/medium/large) that pass threshold
                    for mask_idx, score in enumerate(point_iou_scores):
                        if score > cfg.sam_pred_iou_thresh:
                            mask = masks[point_idx, mask_idx].numpy()
                            all_masks.append(mask)
                else:
                    # Only use the best scoring mask per point (old behavior)
                    best_idx = np.argmax(point_iou_scores)
                    if point_iou_scores[best_idx] > cfg.sam_pred_iou_thresh:
                        mask = masks[point_idx, best_idx].numpy()
                        all_masks.append(mask)

    # Extract boundaries from SAM masks
    if len(all_masks) > 0:
        all_masks = np.array(all_masks)
        boundary_map = _extract_boundaries_from_masks(all_masks)
        if boundary_map is not None:
            boundary_map = _morph_close((boundary_map * 255).astype(np.uint8), cfg.morph_kernel)
            boundary_map = (boundary_map > 0).astype(np.float32)
        else:
            boundary_map = np.zeros((h, w), dtype=np.float32)
    else:
        boundary_map = np.zeros((h, w), dtype=np.float32)

    # Optionally combine with Canny edges to catch what SAM misses (like glass edges)
    if cfg.sam_combine_with_canny:
        canny_edges = rgb_edges(rgb, cfg)
        boundary_map = np.maximum(boundary_map, canny_edges)

    # Fallback to pure Canny if SAM produced nothing
    if boundary_map.sum() == 0:
        return rgb_edges(rgb, cfg)

    return boundary_map


def sam_edges_depth(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Use SAM to generate boundary/edge map from depth image.

    Uses robust preprocessing: sanitization + inverse depth for better edges.
    Collects all 3 mask scales (small/medium/large) per point.
    Optionally combines with Canny depth edges for complete coverage.

    NOTE: For transparent objects, the depth sensor sees THROUGH the glass,
    so depth edges will show the background geometry, not the glass itself.
    This is expected behavior - we use this to find what's behind the glass.
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

            iou_scores = outputs.iou_scores.cpu().numpy()

            # Iterate over each point in the batch
            for point_idx in range(masks.shape[0]):
                point_iou_scores = iou_scores[0, point_idx]  # 3 scores for 3 mask scales

                if cfg.sam_use_all_masks:
                    # Collect ALL 3 mask scales that pass threshold
                    for mask_idx, score in enumerate(point_iou_scores):
                        if score > cfg.sam_pred_iou_thresh:
                            mask = masks[point_idx, mask_idx].numpy()
                            all_masks.append(mask)
                else:
                    # Only use the best scoring mask per point
                    best_idx = np.argmax(point_iou_scores)
                    if point_iou_scores[best_idx] > cfg.sam_pred_iou_thresh:
                        mask = masks[point_idx, best_idx].numpy()
                        all_masks.append(mask)

    # Extract boundaries from SAM masks
    if len(all_masks) > 0:
        all_masks = np.array(all_masks)
        boundary_map = _extract_boundaries_from_masks(all_masks)
        if boundary_map is not None:
            boundary_map = _morph_close((boundary_map * 255).astype(np.uint8), cfg.morph_kernel)
            boundary_map = (boundary_map > 0).astype(np.float32)
        else:
            boundary_map = np.zeros((h, w), dtype=np.float32)
    else:
        boundary_map = np.zeros((h, w), dtype=np.float32)

    # Optionally combine with Canny edges to catch depth discontinuities SAM misses
    if cfg.sam_combine_with_canny:
        canny_edges = depth_edges(depth, cfg)
        boundary_map = np.maximum(boundary_map, canny_edges)

    # Fallback to pure Canny if SAM produced nothing
    if boundary_map.sum() == 0:
        return depth_edges(depth, cfg)

    return boundary_map


# ============================================================================
# Classic edge detection (fallback when SAM is disabled)
# ============================================================================

def rgb_edges(rgb_bgr: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Enhanced edge detection on RGB images, optimized for transparent objects.

    Transparent objects (glass, plastic) are challenging because:
    1. Low contrast - subtle intensity differences
    2. No texture - smooth surfaces lack strong gradients
    3. Specular highlights - reflections can confuse detectors

    This enhanced version uses multiple techniques:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
    2. Multi-channel processing (grayscale, LAB L-channel, HSV saturation)
    3. Multi-scale Canny (high + low thresholds combined)
    4. Gradient magnitude thresholding as additional edge source

    Args:
        rgb_bgr: (H, W, 3) BGR image
        cfg: GuidanceConfig with edge detection parameters

    Returns:
        (H, W) binary edge map as float32 in [0, 1]
    """
    if not cfg.rgb_use_enhanced:
        # Original simple Canny
        if rgb_bgr.ndim == 3 and rgb_bgr.shape[2] == 3:
            gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_bgr
        edges = cv2.Canny(gray, cfg.rgb_canny_thresh1, cfg.rgb_canny_thresh2,
                          apertureSize=cfg.rgb_canny_aperture)
        edges = _morph_close(edges, cfg.morph_kernel)
        return (edges > 0).astype(np.float32)

    # =========================================================================
    # Enhanced edge detection for transparent objects
    # =========================================================================
    h, w = rgb_bgr.shape[:2]
    edge_maps = []

    # Convert to grayscale
    if rgb_bgr.ndim == 3 and rgb_bgr.shape[2] == 3:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_bgr.copy()

    # ---------------------------------------------------------------------------
    # 1. CLAHE-enhanced grayscale edges
    # CLAHE enhances local contrast, making subtle glass boundaries more visible
    # ---------------------------------------------------------------------------
    if cfg.rgb_enhance_clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.rgb_clahe_clip,
                                 tileGridSize=(cfg.rgb_clahe_grid, cfg.rgb_clahe_grid))
        gray_clahe = clahe.apply(gray)
    else:
        gray_clahe = gray

    # Standard threshold Canny on CLAHE-enhanced grayscale
    edges_gray = cv2.Canny(gray_clahe, cfg.rgb_canny_thresh1, cfg.rgb_canny_thresh2,
                           apertureSize=cfg.rgb_canny_aperture)
    edge_maps.append(edges_gray)

    # ---------------------------------------------------------------------------
    # 2. Multi-scale Canny (catch subtle edges with lower thresholds)
    # Lower thresholds catch faint glass boundaries that standard thresholds miss
    # ---------------------------------------------------------------------------
    if cfg.rgb_multi_scale:
        edges_low = cv2.Canny(gray_clahe, cfg.rgb_canny_low_thresh1, cfg.rgb_canny_low_thresh2,
                              apertureSize=cfg.rgb_canny_aperture)
        edge_maps.append(edges_low)

    # ---------------------------------------------------------------------------
    # 3. LAB L-channel edges (luminance is more robust to color variations)
    # The L-channel in LAB space better captures brightness changes at glass edges
    # ---------------------------------------------------------------------------
    if cfg.rgb_use_lab_channel and rgb_bgr.ndim == 3:
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        if cfg.rgb_enhance_clahe:
            l_channel = clahe.apply(l_channel)

        edges_lab = cv2.Canny(l_channel, cfg.rgb_canny_thresh1, cfg.rgb_canny_thresh2,
                              apertureSize=cfg.rgb_canny_aperture)
        edge_maps.append(edges_lab)

        if cfg.rgb_multi_scale:
            edges_lab_low = cv2.Canny(l_channel, cfg.rgb_canny_low_thresh1, cfg.rgb_canny_low_thresh2,
                                      apertureSize=cfg.rgb_canny_aperture)
            edge_maps.append(edges_lab_low)

    # ---------------------------------------------------------------------------
    # 4. HSV Saturation channel edges
    # Glass often has subtle saturation changes at boundaries (reflections, tints)
    # ---------------------------------------------------------------------------
    if cfg.rgb_use_saturation and rgb_bgr.ndim == 3:
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        if cfg.rgb_enhance_clahe:
            s_channel = clahe.apply(s_channel)

        # Use lower thresholds for saturation (often subtle changes)
        edges_sat = cv2.Canny(s_channel, cfg.rgb_canny_low_thresh1, cfg.rgb_canny_low_thresh2,
                              apertureSize=cfg.rgb_canny_aperture)
        edge_maps.append(edges_sat)

    # ---------------------------------------------------------------------------
    # 5. Gradient magnitude thresholding (catches edges Canny might miss)
    # Direct gradient magnitude is more sensitive to subtle intensity changes
    # ---------------------------------------------------------------------------
    if cfg.rgb_gradient_magnitude:
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray_clahe, (3, 3), 0)

        # Compute Sobel gradients
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

        # Adaptive thresholding based on image statistics
        thresh = np.percentile(mag, cfg.rgb_gradient_percentile)
        edges_grad = (mag > thresh).astype(np.uint8) * 255

        # Thin the gradient edges using morphological skeleton approximation
        kernel = np.ones((2, 2), np.uint8)
        edges_grad = cv2.erode(edges_grad, kernel, iterations=1)

        edge_maps.append(edges_grad)    # ---------------------------------------------------------------------------
    # Combine all edge maps (union)
    # ---------------------------------------------------------------------------
    combined = np.zeros((h, w), dtype=np.uint8)
    for em in edge_maps:
        combined = np.maximum(combined, em)

    # Morphological closing to connect nearby edges
    combined = _morph_close(combined, cfg.morph_kernel)

    return (combined > 0).astype(np.float32)


def depth_edges(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """
    Enhanced edge detection on depth maps.

    Handles the challenges of raw EXR depth data:
    1. Sanitizes NaN/Inf values (sensor errors, sky)
    2. Optionally converts to inverse depth (1/z) for better near-object edges
    3. Normalizes to 0-255 range for edge detection
    4. Uses enhanced techniques (CLAHE, multi-scale) for better edge detection

    Enhanced mode adds:
    - CLAHE for local contrast enhancement (reveals subtle depth discontinuities)
    - Multi-scale Canny (high + low thresholds combined)
    - Optional gradient magnitude thresholding

    Args:
        depth: (H, W) raw depth map (can contain NaN, Inf, large values)
        cfg: GuidanceConfig with depth preprocessing parameters

    Returns:
        (H, W) binary edge map as float32 in [0, 1]
    """
    d = depth.astype(np.float32)
    h, w = d.shape[:2]

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

    # =========================================================================
    # Enhanced edge detection for depth (similar to RGB enhancement)
    # =========================================================================
    if cfg.depth_use_enhanced:
        edge_maps = []

        # Apply CLAHE for local contrast enhancement
        if cfg.depth_enhance_clahe:
            clahe = cv2.createCLAHE(clipLimit=cfg.depth_clahe_clip,
                                     tileGridSize=(cfg.depth_clahe_grid, cfg.depth_clahe_grid))
            d_clahe = clahe.apply(d_uint8)
        else:
            d_clahe = d_uint8

        # Standard threshold Canny on CLAHE-enhanced depth
        if cfg.depth_use_canny:
            edges_std = cv2.Canny(d_clahe, cfg.depth_canny_thresh1, cfg.depth_canny_thresh2,
                                  apertureSize=cfg.rgb_canny_aperture)
            edge_maps.append(edges_std)

        # Multi-scale Canny (lower thresholds to catch subtle depth discontinuities)
        if cfg.depth_multi_scale:
            edges_low = cv2.Canny(d_clahe, cfg.depth_canny_low_thresh1, cfg.depth_canny_low_thresh2,
                                  apertureSize=cfg.rgb_canny_aperture)
            edge_maps.append(edges_low)

        # Gradient magnitude thresholding
        if cfg.depth_gradient_magnitude:
            blurred = cv2.GaussianBlur(d_clahe, (3, 3), 0)
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            thresh = np.percentile(mag, cfg.depth_gradient_percentile)
            edges_grad = (mag > thresh).astype(np.uint8) * 255
            # Thin edges
            kernel = np.ones((2, 2), np.uint8)
            edges_grad = cv2.erode(edges_grad, kernel, iterations=1)
            edge_maps.append(edges_grad)

        # Combine all edge maps
        edges = np.zeros((h, w), dtype=np.uint8)
        for em in edge_maps:
            edges = np.maximum(edges, em)

    else:
        # Original simple edge detection
        if cfg.depth_use_canny:
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

def _get_rgb_edges(rgb_bgr: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """Get RGB edges using the configured edge detector."""
    if cfg.edge_detector == 'pidinet':
        try:
            from pidinet_edges import pidinet_edges_rgb, PiDiNetConfig
            # Only pass model_path if explicitly set, otherwise use PiDiNetConfig default
            pidi_kwargs = dict(
                device=cfg.pidinet_device,
                dilation=cfg.pidinet_dilation,
                threshold=cfg.pidinet_threshold,
                morph_erode=cfg.pidinet_morph_erode,
                depth_max_distance=cfg.depth_max_distance,
                depth_use_inverse=cfg.depth_use_inverse,
            )
            if cfg.pidinet_weights_path is not None:
                pidi_kwargs['model_path'] = cfg.pidinet_weights_path
            pidi_cfg = PiDiNetConfig(**pidi_kwargs)
            return pidinet_edges_rgb(rgb_bgr, pidi_cfg)
        except ImportError:
            print("WARNING: PiDiNet not available, falling back to Canny")
            return rgb_edges(rgb_bgr, cfg)
    elif cfg.edge_detector == 'sam' or (cfg.use_sam and cfg.use_sam_rgb):
        return sam_edges_rgb(rgb_bgr, cfg)
    else:
        return rgb_edges(rgb_bgr, cfg)


def _get_depth_edges(depth: np.ndarray, cfg: GuidanceConfig) -> np.ndarray:
    """Get depth edges using the configured edge detector."""
    if cfg.edge_detector == 'pidinet':
        try:
            from pidinet_edges import pidinet_edges_depth, PiDiNetConfig
            # Only pass model_path if explicitly set, otherwise use PiDiNetConfig default
            pidi_kwargs = dict(
                device=cfg.pidinet_device,
                dilation=cfg.pidinet_dilation,
                threshold=cfg.pidinet_threshold,
                morph_erode=cfg.pidinet_morph_erode,
                depth_max_distance=cfg.depth_max_distance,
                depth_use_inverse=cfg.depth_use_inverse,
            )
            if cfg.pidinet_weights_path is not None:
                pidi_kwargs['model_path'] = cfg.pidinet_weights_path
            pidi_cfg = PiDiNetConfig(**pidi_kwargs)
            return pidinet_edges_depth(depth, pidi_cfg)
        except ImportError:
            print("WARNING: PiDiNet not available, falling back to Canny")
            return depth_edges(depth, cfg)
    elif cfg.edge_detector == 'sam' or (cfg.use_sam and cfg.use_sam_depth):
        return sam_edges_depth(depth, cfg)
    else:
        return depth_edges(depth, cfg)


def m_da(rgb_bgr: np.ndarray, depth: np.ndarray, cfg: GuidanceConfig,
         mask: Optional[np.ndarray] = None,
         zeroed_depth: Optional[np.ndarray] = None,
         fallback_threshold: float = 0.01) -> np.ndarray:
    """
    Depth-Aware boundary map for OPTICAL inpainting (Track A).

    Per DITR paper: M_DA = M_RGB \\\\ C_U(M_D) = M_RGB ∩ M_D (INTERSECTION)

    - M_RGB: Edges from RGB image (sees transparent objects + background)
    - M_D: Edges from depth image (sees background THROUGH transparent objects)
    - M_DA: Difference (M_RGB - M_D) = edges visible in RGB but NOT in depth
            This captures glass boundaries that the depth sensor cannot see.

    IMPORTANT: For depth edge detection, use `zeroed_depth` (depth with transparent
    region set to 0) to get edges of what's visible through/behind the glass.

    Safety Net: If M_DA < 1% of mask pixels, fallback to M_RGB (edges inside mask).
    This prevents guidance map being empty/useless.

    Args:
        rgb_bgr: (H, W, 3) BGR image
        depth: (H, W) raw depth map (used if zeroed_depth not provided)
        cfg: GuidanceConfig
        mask: Optional (H, W) mask where 1=transparent area
        zeroed_depth: Optional (H, W) depth with transparent region zeroed out.
                      If provided, this is used for depth edge detection.
        fallback_threshold: Minimum ratio of M_DA pixels to mask pixels (default 1%)

    Returns:
        (H, W) guidance map - difference of RGB and Depth edges, masked to transparent region
    """
    # Get RGB edges (sees everything including transparent object boundaries)
    e_rgb = _get_rgb_edges(rgb_bgr, cfg)

    # Get Depth edges from zeroed depth (sees background through glass)
    # Use zeroed_depth if provided, otherwise use raw depth
    depth_for_edges = zeroed_depth if zeroed_depth is not None else depth
    e_d = _get_depth_edges(depth_for_edges, cfg)

    # M_DA = M_RGB - M_D (DIFFERENCE)
    # This captures edges visible in RGB but NOT in depth (glass boundaries)
    # The depth sensor sees through glass, so depth edges show background geometry
    # RGB edges see the glass itself, so (RGB - Depth) = glass boundaries only
    mda_raw = np.maximum(e_rgb - e_d, 0)  # Difference, clipped to non-negative

    # Apply mask: guidance only inside transparent region
    if mask is not None:
        mda_masked = mda_raw * mask
        e_rgb_masked = e_rgb * mask

        # =====================================================================
        # SAFETY NET: Fallback if difference is too sparse
        # If M_DA has less than threshold % of mask pixels, fallback to M_RGB
        # =====================================================================
        mask_pixel_count = mask.sum()
        mda_pixel_count = mda_masked.sum()

        if mask_pixel_count > 0:
            mda_ratio = mda_pixel_count / mask_pixel_count

            if mda_ratio < fallback_threshold:
                # Difference is too sparse, fallback to M_RGB inside mask
                e_rgb_count = e_rgb_masked.sum()

                if e_rgb_count > 0:
                    out = e_rgb_masked
                    # Optionally log this for debugging
                    # print(f"M_DA fallback: using M_RGB ({e_rgb_count:.0f} px)")
                else:
                    # M_RGB also empty - return zeros
                    out = mda_masked
            else:
                # Difference is sufficient, use it
                out = mda_masked
        else:
            # No mask pixels (shouldn't happen, but handle gracefully)
            out = mda_masked
    else:
        # No mask provided, return raw difference
        out = mda_raw

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
    # Get RGB edges using configured detector
    edges = _get_rgb_edges(rgb_bgr, cfg)

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
