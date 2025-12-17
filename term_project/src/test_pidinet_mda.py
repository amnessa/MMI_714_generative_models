"""
Test script to visualize M_DA extraction using PiDiNet edge detector.

This script compares edge detection methods (Canny vs PiDiNet) and shows
how M_DA (intersection of RGB and Depth edges) is computed.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from guidance_maps import GuidanceConfig, m_da, m_rgb, rgb_edges, depth_edges
from pidinet_edges import (
    PiDiNetConfig, pidinet_edges_rgb, pidinet_edges_depth,
    compute_mda_pidinet, compute_mrgb_pidinet
)


def apply_safety_net(e_rgb, e_depth, mask, fallback_threshold=0.01):
    """
    Apply safety net logic: if intersection is too sparse, fallback to M_RGB or M_D.

    Returns:
        (mda, fallback_used): M_DA result and which fallback was used (None, 'rgb', or 'depth')
    """
    e_rgb_masked = e_rgb * mask
    e_d_masked = e_depth * mask
    intersection = np.minimum(e_rgb, e_depth) * mask

    mask_pixels = mask.sum()
    if mask_pixels == 0:
        return intersection, None

    ratio = intersection.sum() / mask_pixels

    if ratio < fallback_threshold:
        e_rgb_count = e_rgb_masked.sum()
        e_d_count = e_d_masked.sum()

        if e_rgb_count > 0 or e_d_count > 0:
            if e_rgb_count >= e_d_count:
                return e_rgb_masked, 'rgb'
            else:
                return e_d_masked, 'depth'
        else:
            return intersection, 'none'

    return intersection, None

# Configuration
DATASET_ROOT = "/home/cago/MMI_714_generative_models/term_project/dataset/cleargrasp-dataset-train-resized/"
WEIGHTS_PATH = "/home/cago/MMI_714_generative_models/term_project/src/weights/table5_pidinet.pth"
IMG_SIZE = (128, 128)


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
    rgb = cv2.resize(rgb, IMG_SIZE)

    # Load depth
    depth_name = fname.replace("-rgb.jpg", "-depth-rectified.exr")
    depth_path = os.path.join(depth_dir, depth_name)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = cv2.resize(depth, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # Load mask
    mask_name = fname.replace("-rgb.jpg", "-segmentation-mask.png")
    mask_path = os.path.join(mask_dir, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)

    return rgb, depth, mask, fname


def compare_edge_detectors(rgb, depth, mask, title=""):
    """Compare Canny vs PiDiNet edge detection and M_DA computation."""

    print(f"\n{'='*60}")
    print(f"Testing: {title}")
    print(f"{'='*60}")

    # Setup configs
    canny_cfg = GuidanceConfig(
        edge_detector='canny',
        depth_max_distance=10.0,
        depth_use_inverse=True,
    )

    pidinet_cfg = PiDiNetConfig(
        model_path=WEIGHTS_PATH,
        device="cuda",
        threshold=0.5,
        depth_max_distance=10.0,
        depth_use_inverse=True,
    )

    # Canny edges (using zeroed depth for depth edges)
    print("Computing Canny edges...")
    e_rgb_canny = rgb_edges(rgb, canny_cfg)
    zeroed_depth = depth * (1.0 - mask)  # Zero out transparent region
    e_depth_canny = depth_edges(zeroed_depth, canny_cfg)
    mda_canny_raw = np.minimum(e_rgb_canny, e_depth_canny) * mask
    mda_canny, canny_fallback = apply_safety_net(e_rgb_canny, e_depth_canny, mask)
    mrgb_canny = e_rgb_canny * (1.0 - mask)

    # PiDiNet edges (using zeroed depth for depth edges)
    print("Computing PiDiNet edges...")
    e_rgb_pidi = pidinet_edges_rgb(rgb, pidinet_cfg)
    e_depth_pidi = pidinet_edges_depth(zeroed_depth, pidinet_cfg)
    mda_pidi_raw = np.minimum(e_rgb_pidi, e_depth_pidi) * mask
    mda_pidi, pidi_fallback = apply_safety_net(e_rgb_pidi, e_depth_pidi, mask)
    mrgb_pidi = e_rgb_pidi * (1.0 - mask)

    # Stats
    print(f"\nEdge detection stats:")
    print(f"  Mask pixels: {mask.sum():.0f}")
    print()
    print(f"  Canny RGB edges:    {e_rgb_canny.sum():.0f} px (inside mask: {(e_rgb_canny * mask).sum():.0f})")
    print(f"  Canny Depth edges:  {e_depth_canny.sum():.0f} px (inside mask: {(e_depth_canny * mask).sum():.0f})")
    print(f"  Canny Intersection: {mda_canny_raw.sum():.0f} px")
    print(f"  Canny M_DA:         {mda_canny.sum():.0f} px (fallback: {canny_fallback})")
    print(f"  Canny M_RGB:        {mrgb_canny.sum():.0f} px (RGB × background)")
    print()
    print(f"  PiDiNet RGB edges:  {e_rgb_pidi.sum():.0f} px (inside mask: {(e_rgb_pidi * mask).sum():.0f})")
    print(f"  PiDiNet Depth edges:{e_depth_pidi.sum():.0f} px (inside mask: {(e_depth_pidi * mask).sum():.0f})")
    print(f"  PiDiNet Intersection: {mda_pidi_raw.sum():.0f} px")
    print(f"  PiDiNet M_DA:       {mda_pidi.sum():.0f} px (fallback: {pidi_fallback})")
    print(f"  PiDiNet M_RGB:      {mrgb_pidi.sum():.0f} px (RGB × background)")

    # Visualize
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f"Edge Detection Comparison: {title}", fontsize=14)

    # Row 1: Input data
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis('off')

    d_viz = np.clip(depth, 0, 10)
    im = axes[0, 1].imshow(d_viz, cmap='viridis')
    axes[0, 1].set_title("Raw Depth")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    d_inv = 1.0 / (np.clip(depth, 0.001, 10) + 0.001)
    im = axes[0, 2].imshow(d_inv, cmap='magma')
    axes[0, 2].set_title("Inverse Depth (1/z)")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[0, 3].imshow(mask, cmap='gray')
    axes[0, 3].set_title(f"Mask (glass={mask.sum():.0f}px)")
    axes[0, 3].axis('off')

    # Simulate zeroed depth (as per new data pipeline)
    zeroed_depth = depth * (1.0 - mask)
    im = axes[0, 4].imshow(zeroed_depth, cmap='viridis')
    axes[0, 4].set_title("Zeroed Depth (new pipeline)")
    axes[0, 4].axis('off')
    plt.colorbar(im, ax=axes[0, 4], fraction=0.046)

    # Row 2: Canny edges
    axes[1, 0].imshow(e_rgb_canny, cmap='gray')
    axes[1, 0].set_title(f"Canny RGB ({e_rgb_canny.sum():.0f}px)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(e_depth_canny, cmap='gray')
    axes[1, 1].set_title(f"Canny Depth ({e_depth_canny.sum():.0f}px)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.minimum(e_rgb_canny, e_depth_canny), cmap='hot')
    axes[1, 2].set_title(f"Canny RGB∩Depth\n({np.minimum(e_rgb_canny, e_depth_canny).sum():.0f}px)")
    axes[1, 2].axis('off')

    axes[1, 3].imshow(mda_canny, cmap='hot')
    axes[1, 3].set_title(f"Canny M_DA\n({mda_canny.sum():.0f}px)")
    axes[1, 3].axis('off')

    axes[1, 4].imshow(mrgb_canny, cmap='hot')
    axes[1, 4].set_title(f"Canny M_RGB\n({mrgb_canny.sum():.0f}px)")
    axes[1, 4].axis('off')

    # Row 3: PiDiNet edges
    axes[2, 0].imshow(e_rgb_pidi, cmap='gray')
    axes[2, 0].set_title(f"PiDiNet RGB ({e_rgb_pidi.sum():.0f}px)")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(e_depth_pidi, cmap='gray')
    axes[2, 1].set_title(f"PiDiNet Depth ({e_depth_pidi.sum():.0f}px)")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(np.minimum(e_rgb_pidi, e_depth_pidi), cmap='hot')
    axes[2, 2].set_title(f"PiDiNet RGB∩Depth\n({np.minimum(e_rgb_pidi, e_depth_pidi).sum():.0f}px)")
    axes[2, 2].axis('off')

    axes[2, 3].imshow(mda_pidi, cmap='hot')
    axes[2, 3].set_title(f"PiDiNet M_DA\n({mda_pidi.sum():.0f}px)")
    axes[2, 3].axis('off')

    axes[2, 4].imshow(mrgb_pidi, cmap='hot')
    axes[2, 4].set_title(f"PiDiNet M_RGB\n({mrgb_pidi.sum():.0f}px)")
    axes[2, 4].axis('off')

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("M_DA Extraction Test with PiDiNet")
    print("=" * 60)

    # Check if weights exist
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
        print("Run: python -c \"from pidinet_edges import download_pidinet_weights; download_pidinet_weights('weights')\"")
        return

    # Get list of classes
    class_folders = [
        f for f in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, f))
        and not f.startswith(".")
        and not f.startswith("guidance")
    ]
    print(f"Found {len(class_folders)} classes: {class_folders[:3]}...")

    # Test on a few samples from different classes
    test_cases = [
        ("cup-with-waves-train", 8),
        ("cup-with-waves-train", 4589),
        ("stemless-plastic-champagne-glass-train", 149),
    ]

    figs = []
    for class_name, idx in test_cases:
        if class_name in class_folders:
            try:
                rgb, depth, mask, fname = load_sample(class_name, idx)
                fig = compare_edge_detectors(rgb, depth, mask, f"{class_name}/{fname}")
                figs.append(fig)
            except Exception as e:
                print(f"Error loading {class_name}[{idx}]: {e}")

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
