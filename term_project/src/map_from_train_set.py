import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm
from guidance_maps import m_da, m_rgb, GuidanceConfig

# Configuration
DATASET_ROOT = "/home/cago/MMI_714_generative_models/term_project/dataset/cleargrasp-dataset-train-resized/"
WEIGHTS_PATH = "/home/cago/MMI_714_generative_models/term_project/src/weights/table5_pidinet.pth"

# Output folders (will be created)
GUIDANCE_DIR_OPT = "guidance-optical-mda"
GUIDANCE_DIR_GEO = "guidance-geometric-mrgb"

# Image size
IMG_SIZE = (128, 128)


def process_dataset(edge_detector: str = 'pidinet'):
    """
    Pre-compute guidance maps for the entire dataset.

    Args:
        edge_detector: 'pidinet', 'canny', or 'sam'
    """
    # Setup config
    cfg = GuidanceConfig(
        edge_detector=edge_detector,
        # PiDiNet settings
        pidinet_weights_path=WEIGHTS_PATH if edge_detector == 'pidinet' else None,
        pidinet_device="cuda",
        pidinet_threshold=0.5,
        # SAM settings (for edge_detector='sam')
        use_sam=edge_detector == 'sam',
        use_sam_rgb=True,
        use_sam_depth=True,
        sam_device="cuda",
        sam_model_name="facebook/sam-vit-base",
        # Robust depth settings (used for all methods)
        depth_max_distance=10.0,
        depth_use_inverse=True,
        depth_use_canny=True,
        depth_canny_thresh1=30,
        depth_canny_thresh2=100,
        depth_edge_dilate=0,
    )

    print(f"Edge detector: {edge_detector}")
    print(f"Depth preprocessing: inverse_depth={cfg.depth_use_inverse}")

    # Walk through dataset - exclude hidden and output folders
    class_folders = [
        f for f in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, f))
        and not f.startswith(".")
        and not f.startswith("guidance")
    ]

    print(f"Found {len(class_folders)} classes: {class_folders}")

    total_processed = 0
    total_skipped = 0

    for class_name in class_folders:
        class_path = os.path.join(DATASET_ROOT, class_name)

        # Create output directories inside each class folder
        save_dir_opt = os.path.join(class_path, GUIDANCE_DIR_OPT)
        save_dir_geo = os.path.join(class_path, GUIDANCE_DIR_GEO)
        os.makedirs(save_dir_opt, exist_ok=True)
        os.makedirs(save_dir_geo, exist_ok=True)

        # Get list of images
        rgb_dir = os.path.join(class_path, "rgb-imgs")
        depth_dir = os.path.join(class_path, "depth-imgs-rectified")
        mask_dir = os.path.join(class_path, "segmentation-masks")

        if not os.path.exists(rgb_dir):
            print(f"  Skipping {class_name}: no rgb-imgs folder")
            continue

        filenames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])

        print(f"\nProcessing {class_name}: {len(filenames)} images")

        for fname in tqdm(filenames, desc=f"  {class_name}", leave=False):
            # Check if already processed
            save_name = fname.replace("-rgb.jpg", ".png")
            if os.path.exists(os.path.join(save_dir_opt, save_name)):
                total_skipped += 1
                continue

            # 1. Read Inputs
            rgb_path = os.path.join(rgb_dir, fname)
            depth_name = fname.replace("-rgb.jpg", "-depth-rectified.exr")
            mask_name = fname.replace("-rgb.jpg", "-segmentation-mask.png")

            depth_path = os.path.join(depth_dir, depth_name)
            mask_path = os.path.join(mask_dir, mask_name)

            if not (os.path.exists(depth_path) and os.path.exists(mask_path)):
                total_skipped += 1
                continue

            try:
                # Load data
                rgb = cv2.imread(rgb_path)
                rgb = cv2.resize(rgb, IMG_SIZE)

                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    total_skipped += 1
                    continue
                if depth.ndim == 3:
                    depth = depth[..., 0]
                depth = cv2.resize(depth, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)

                # Create zeroed depth (transparent region set to 0)
                # This is what the depth sensor "sees through" the glass
                zeroed_depth = depth * (1.0 - mask)

                # 2. Compute Guidance Maps
                # M_DA for Optical (Masked to glass)
                # Uses zeroed_depth for depth edges to see background through glass
                guidance_opt = m_da(rgb, depth, cfg, mask, zeroed_depth=zeroed_depth)

                # M_RGB for Geometric (Masked to background)
                guidance_geo = m_rgb(rgb, cfg, mask)

                # 3. Save as Images (uint8 0-255)
                cv2.imwrite(os.path.join(save_dir_opt, save_name),
                           (guidance_opt * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_dir_geo, save_name),
                           (guidance_geo * 255).astype(np.uint8))

                total_processed += 1

            except Exception as e:
                print(f"\n  Error processing {fname}: {e}")
                total_skipped += 1
                continue

    print(f"\n{'='*60}")
    print(f"Done! Processed: {total_processed}, Skipped: {total_skipped}")


if __name__ == "__main__":
    import argparse
    from typing import Literal

    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-detector", type=str, default="pidinet",
                        choices=["pidinet", "canny", "sam"],
                        help="Edge detector to use: pidinet (default), canny, or sam")
    args = parser.parse_args()

    process_dataset(edge_detector=args.edge_detector)