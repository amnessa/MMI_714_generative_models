"""
Quick test script to visualize depth edge extraction and M_DA computation.
Run this to debug the guidance map generation before processing the full dataset.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from guidance_maps import (
    GuidanceConfig, rgb_edges, depth_edges, m_da, m_rgb,
    sam_edges_rgb, sam_edges_depth
)

# Configuration
DATASET_ROOT = "/home/cago/MMI_714_generative_models/term_project/dataset/cleargrasp-dataset-train-resized/"

def load_sample(class_name: str, idx: int = 0):
    """Load a sample from the dataset."""
    class_path = os.path.join(DATASET_ROOT, class_name)
    rgb_dir = os.path.join(class_path, "rgb-imgs")
    depth_dir = os.path.join(class_path, "depth-imgs-rectified")
    mask_dir = os.path.join(class_path, "segmentation-masks")

    filenames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    fname = filenames[idx]

    # Load RGB
    rgb_path = os.path.join(rgb_dir, fname)
    rgb = cv2.imread(rgb_path)
    rgb = cv2.resize(rgb, (128, 128))

    # Load depth
    depth_name = fname.replace("-rgb.jpg", "-depth-rectified.exr")
    depth_path = os.path.join(depth_dir, depth_name)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = cv2.resize(depth, (128, 128), interpolation=cv2.INTER_NEAREST)

    # Load mask
    mask_name = fname.replace("-rgb.jpg", "-segmentation-mask.png")
    mask_path = os.path.join(mask_dir, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)

    return rgb, depth, mask, fname


def visualize_depth_processing(rgb, depth, mask, cfg: GuidanceConfig, title=""):
    """Visualize the depth edge extraction pipeline with SAM vs Canny comparison."""

    # Raw depth stats
    print(f"\n{'='*60}")
    print(f"Raw depth stats:")
    print(f"  Shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"  Min: {np.nanmin(depth):.4f}, Max: {np.nanmax(depth):.4f}")
    print(f"  NaN count: {np.isnan(depth).sum()}, Inf count: {np.isinf(depth).sum()}")
    print(f"  Valid pixels: {np.isfinite(depth).sum()} / {depth.size}")

    # Always compute Canny edges for comparison
    e_rgb_canny = rgb_edges(rgb, cfg)

    # Create zeroed depth for depth edge detection
    zeroed_depth = depth * (1.0 - mask)
    e_depth_canny = depth_edges(zeroed_depth, cfg)

    # Compute SAM edges if enabled
    if cfg.use_sam and cfg.use_sam_rgb:
        e_rgb_sam = sam_edges_rgb(rgb, cfg)
    else:
        e_rgb_sam = None

    if cfg.use_sam and cfg.use_sam_depth:
        e_depth_sam = sam_edges_depth(zeroed_depth, cfg)
    else:
        e_depth_sam = None

    # Compute M_DA and M_RGB using the m_da function (includes safety net)
    mda = m_da(rgb, depth, cfg, mask, zeroed_depth=zeroed_depth)
    mrgb = m_rgb(rgb, cfg, mask)

    # Determine which edges are actually being used
    e_rgb_used = e_rgb_sam if (cfg.use_sam and cfg.use_sam_rgb) else e_rgb_canny
    e_depth_used = e_depth_sam if (cfg.use_sam and cfg.use_sam_depth) else e_depth_canny

    # Edge overlap analysis (using what's actually used)
    rgb_edge_pixels = e_rgb_used.sum()
    depth_edge_pixels = e_depth_used.sum()
    overlap = (e_rgb_used * e_depth_used).sum()  # Overlap IS M_DA (intersection)

    # M_DA is now intersection: min(RGB, Depth) = overlap in binary case
    mda_raw = np.minimum(e_rgb_used, e_depth_used)

    print(f"\nEdge analysis (method used):")
    print(f"  RGB edges ({'SAM' if e_rgb_sam is not None else 'Canny'}): {rgb_edge_pixels:.0f} pixels")
    print(f"  Depth edges ({'SAM' if e_depth_sam is not None else 'Canny'}): {depth_edge_pixels:.0f} pixels")
    print(f"  Intersection (RGB ∩ Depth): {overlap:.0f} pixels ({100*overlap/max(rgb_edge_pixels,1):.1f}% of RGB edges)")
    print(f"  M_DA raw (before mask): {mda_raw.sum():.0f} pixels")
    print(f"  M_DA (glass region only): {mda.sum():.0f} pixels")
    print(f"  M_RGB (background only): {mrgb.sum():.0f} pixels")

    if e_rgb_sam is not None:
        print(f"\n  Canny RGB edges: {e_rgb_canny.sum():.0f} px vs SAM RGB: {e_rgb_sam.sum():.0f} px")
    if e_depth_sam is not None:
        print(f"  Canny Depth edges: {e_depth_canny.sum():.0f} px vs SAM Depth: {e_depth_sam.sum():.0f} px")

    # Visualize with 3 rows now
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"{title}\nRGB: {'SAM' if e_rgb_sam is not None else 'Canny'}, Depth: {'SAM' if e_depth_sam is not None else 'Canny'}", fontsize=14)

    # Row 1: Inputs
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis('off')

    d_viz = np.clip(depth, 0, cfg.depth_max_distance)
    im = axes[0, 1].imshow(d_viz, cmap='viridis')
    axes[0, 1].set_title(f"Raw Depth")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    d_inv = 1.0 / (np.clip(depth, 0.001, cfg.depth_max_distance) + 0.001)
    im = axes[0, 2].imshow(d_inv, cmap='magma')
    axes[0, 2].set_title("Inverse Depth (1/z)")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[0, 3].imshow(mask, cmap='gray')
    axes[0, 3].set_title(f"Mask (glass={mask.sum():.0f}px)")
    axes[0, 3].axis('off')

    # Row 2: Edge comparison (Canny vs SAM)
    axes[1, 0].imshow(e_rgb_canny, cmap='gray')
    axes[1, 0].set_title(f"RGB Canny ({e_rgb_canny.sum():.0f} px)")
    axes[1, 0].axis('off')

    if e_rgb_sam is not None:
        axes[1, 1].imshow(e_rgb_sam, cmap='gray')
        axes[1, 1].set_title(f"RGB SAM ({e_rgb_sam.sum():.0f} px)")
    else:
        axes[1, 1].imshow(np.zeros_like(e_rgb_canny), cmap='gray')
        axes[1, 1].set_title("RGB SAM (disabled)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(e_depth_canny, cmap='gray')
    axes[1, 2].set_title(f"Depth Canny ({e_depth_canny.sum():.0f} px)")
    axes[1, 2].axis('off')

    if e_depth_sam is not None:
        axes[1, 3].imshow(e_depth_sam, cmap='gray')
        axes[1, 3].set_title(f"Depth SAM ({e_depth_sam.sum():.0f} px)")
    else:
        axes[1, 3].imshow(np.zeros_like(e_depth_canny), cmap='gray')
        axes[1, 3].set_title("Depth SAM (disabled)")
    axes[1, 3].axis('off')

    # Row 3: Guidance maps (what actually gets used)
    axes[2, 0].imshow(e_rgb_used, cmap='gray')
    axes[2, 0].set_title(f"RGB Used ({rgb_edge_pixels:.0f} px)")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(mda_raw, cmap='hot')
    axes[2, 1].set_title(f"M_DA raw (RGB∩Depth)\n({mda_raw.sum():.0f} px)")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(mda, cmap='hot')
    axes[2, 2].set_title(f"M_DA × mask\n(optical, {mda.sum():.0f} px)")
    axes[2, 2].axis('off')

    axes[2, 3].imshow(mrgb, cmap='hot')
    axes[2, 3].set_title(f"M_RGB × (1-mask)\n(geometric, {mrgb.sum():.0f} px)")
    axes[2, 3].axis('off')

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('/', '_')
    plt.show()

    return e_rgb_used, e_depth_used, mda, mrgb


def main():
    parser = argparse.ArgumentParser(description="Test depth edge extraction methods")
    parser.add_argument("--use-sam", action="store_true",
                        help="Use SAM for both RGB and depth edges")
    parser.add_argument("--use-sam-rgb", action="store_true",
                        help="Use SAM for RGB edges only")
    parser.add_argument("--use-sam-depth", action="store_true",
                        help="Use SAM for depth edges only")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base",
                        choices=["facebook/sam-vit-base", "facebook/sam-vit-large"],
                        help="SAM model to use")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to test")
    parser.add_argument("--samples-per-class", type=int, nargs="+", default=[0, 50, 100],
                        help="Sample indices to test per class")
    args = parser.parse_args()

    # If --use-sam is set, enable both RGB and depth SAM
    use_sam_rgb = args.use_sam or args.use_sam_rgb
    use_sam_depth = args.use_sam or args.use_sam_depth
    use_sam = use_sam_rgb or use_sam_depth

    cfg = GuidanceConfig(
        use_sam=use_sam,
        use_sam_rgb=use_sam_rgb,
        use_sam_depth=use_sam_depth,
        sam_device="cuda" if use_sam else "cpu",
        sam_model_name=args.sam_model,
        depth_max_distance=10.0,
        depth_use_inverse=True,
        depth_use_canny=True,
        depth_canny_thresh1=30,
        depth_canny_thresh2=100,
    )

    print("Testing edge extraction...")
    print(f"Config: use_sam_rgb={use_sam_rgb}, use_sam_depth={use_sam_depth}")
    print(f"        sam_model={args.sam_model if use_sam else 'N/A'}")
    print(f"        inverse_depth={cfg.depth_use_inverse}, canny={cfg.depth_use_canny}")

    # Get list of class folders
    class_folders = [f for f in os.listdir(DATASET_ROOT)
                     if os.path.isdir(os.path.join(DATASET_ROOT, f))
                     and not f.startswith("guidance")]

    print(f"Found {len(class_folders)} classes: {class_folders}")

    # Test on samples
    for class_name in class_folders[:args.num_classes]:
        print(f"\n{'='*60}")
        print(f"Testing class: {class_name}")

        for idx in args.samples_per_class:
            try:
                rgb, depth, mask, fname = load_sample(class_name, idx)
                visualize_depth_processing(
                    rgb, depth, mask, cfg,
                    title=f"{class_name} - {fname}"
                )
            except Exception as e:
                print(f"  Sample {idx} failed: {e}")
                continue


if __name__ == "__main__":
    main()
