import os
import imageio.v3 as iio
import numpy as np
import cv2
from PIL import Image
OPENCV_IO_ENABLE_OPENEXR = True

ROOT_DIR = "."          # current directory
TARGET_SIZE = (256, 256)

IMAGE_EXTS_PIL = {".jpg", ".jpeg", ".png"}
IMAGE_EXTS_EXR = {".exr"}


def resize_exr(path):
    try:
        img = iio.imread(path)  # float32 depth
    except Exception as e:
        print(f"Failed to read EXR with imageio: {path} ({e})")
        return

    if img is None:
        print(f"Failed to read EXR: {path}")
        return

    h, w = img.shape[:2]
    if (w, h) == TARGET_SIZE:
        return

    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    iio.imwrite(path, resized.astype(np.float32))
    print(f"Resized EXR: {path}")


def resize_pil(path):
    try:
        with Image.open(path) as img:
            if img.size == TARGET_SIZE:
                return
            # bilinear for natural images / masks; if you want strict mask resizing use NEAREST
            if img.mode in ("1", "P"):
                # likely segmentation mask; use nearest neighbor to avoid label interpolation
                resample = Image.NEAREST
            else:
                resample = Image.BILINEAR

            img_resized = img.resize(TARGET_SIZE, resample=resample)
            img_resized.save(path)
            print(f"Resized: {path}")
    except Exception as e:
        print(f"Error resizing {path}: {e}")


for root, dirs, files in os.walk(ROOT_DIR):
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(root, fname)

        if ext in IMAGE_EXTS_EXR:
            resize_exr(fpath)
        elif ext in IMAGE_EXTS_PIL:
            resize_pil(fpath)