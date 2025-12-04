"""
Quick test script to visualize depth edge extraction and M_DA computation.
Run this to debug the guidance map generation before processing the full dataset.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from guidance_maps import (
    GuidanceConfig, rgb_edges, depth_edges, m_da, m_rgb
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
    """Visualize the depth edge extraction pipeline."""

    # Raw depth stats
    print(f"\n{'='*60}")
    print(f"Raw depth stats:")
    print(f"  Shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"  Min: {np.nanmin(depth):.4f}, Max: {np.nanmax(depth):.4f}")
    print(f"  NaN count: {np.isnan(depth).sum()}, Inf count: {np.isinf(depth).sum()}")
    print(f"  Valid pixels: {np.isfinite(depth).sum()} / {depth.size}")

    # Compute edges
    e_rgb = rgb_edges(rgb, cfg)
    e_depth = depth_edges(depth, cfg)

    # Compute M_DA
    mda = m_da(rgb, depth, cfg, mask)
    mrgb = m_rgb(rgb, cfg, mask)

    # Edge overlap analysis
    rgb_edge_pixels = e_rgb.sum()
    depth_edge_pixels = e_depth.sum()
    overlap = (e_rgb * e_depth).sum()

    print(f"\nEdge analysis:")
    print(f"  RGB edges: {rgb_edge_pixels:.0f} pixels")
    print(f"  Depth edges: {depth_edge_pixels:.0f} pixels")
    print(f"  Overlap: {overlap:.0f} pixels ({100*overlap/max(rgb_edge_pixels,1):.1f}% of RGB edges)")
    print(f"  M_DA (glass): {mda.sum():.0f} pixels")
    print(f"  M_RGB (background): {mrgb.sum():.0f} pixels")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=14)

    # Row 1: Inputs
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis('off')

    # Show raw depth (clipped for visualization)
    d_viz = np.clip(depth, 0, cfg.depth_max_distance)
    im = axes[0, 1].imshow(d_viz, cmap='viridis')
    axes[0, 1].set_title(f"Raw Depth (clipped to {cfg.depth_max_distance}m)")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Show inverse depth
    d_inv = 1.0 / (np.clip(depth, 0.001, cfg.depth_max_distance) + 0.001)
    im = axes[0, 2].imshow(d_inv, cmap='magma')
    axes[0, 2].set_title("Inverse Depth (1/z)")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[0, 3].imshow(mask, cmap='gray')
    axes[0, 3].set_title("Mask (1=glass)")
    axes[0, 3].axis('off')

    # Row 2: Edges and guidance
    axes[1, 0].imshow(e_rgb, cmap='gray')
    axes[1, 0].set_title(f"RGB Edges ({rgb_edge_pixels:.0f} px)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(e_depth, cmap='gray')
    axes[1, 1].set_title(f"Depth Edges ({depth_edge_pixels:.0f} px)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mda, cmap='hot')
    axes[1, 2].set_title(f"M_DA = RGB - Depth\n(optical guidance, {mda.sum():.0f} px)")
    axes[1, 2].axis('off')

    axes[1, 3].imshow(mrgb, cmap='hot')
    axes[1, 3].set_title(f"M_RGB\n(geometric guidance, {mrgb.sum():.0f} px)")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(f"debug_depth_edges_{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

    return e_rgb, e_depth, mda, mrgb


def main():
    # Use classic edge detection (Canny) - no SAM
    cfg = GuidanceConfig(
        use_sam=False,
        depth_max_distance=10.0,
        depth_use_inverse=True,
        depth_use_canny=True,
        depth_canny_thresh1=30,
        depth_canny_thresh2=100,
    )

    print("Testing robust depth edge extraction...")
    print(f"Config: inverse={cfg.depth_use_inverse}, canny={cfg.depth_use_canny}")

    # Get list of class folders
    class_folders = [f for f in os.listdir(DATASET_ROOT)
                     if os.path.isdir(os.path.join(DATASET_ROOT, f))
                     and not f.startswith("guidance")]

    print(f"Found {len(class_folders)} classes: {class_folders}")

    # Test on a few samples from different classes
    for class_name in class_folders[:2]:  # Test first 2 classes
        print(f"\n{'='*60}")
        print(f"Testing class: {class_name}")

        for idx in [0, 50, 100]:  # Test 3 samples per class
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
