import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .guidance_maps import GuidanceConfig, m_da, m_rgb
except Exception:  # fallback for running as a simple module from notebook
    from guidance_maps import GuidanceConfig, m_da, m_rgb


IMG_EXTS = (".png", ".jpg", ".jpeg")
DEPTH_EXTS = (".exr", ".png", ".npy")
MASK_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class DepthDataConfig:
    root: str
    rgb_dir: str = "rgb-imgs"
    depth_dir: str = "depth-imgs-rectified"
    mask_dir: str = "segmentation-masks"  # binary mask for transparent areas (1=transparent)
    guidance_optical_dir: str = "guidance-optical-mda"  # Pre-computed M_DA maps
    guidance_geometric_dir: str = "guidance-geometric-mrgb"  # Pre-computed M_RGB maps
    size: Tuple[int, int] = (128, 128)
    depth_minmax: Tuple[float, float] = (0.0, 3.0)
    # noise parameters in normalized depth space [-1, 1]
    sigma_mask: float = 0.08   # heavy noise in transparent region
    sigma_global: float = 0.01 # light noise overall
    branch: Literal["optical", "geometric"] = "optical"
    include_guidance: bool = False  # Disabled for ablation study
    use_precomputed_guidance: bool = True  # Use pre-computed maps if available



def _read_rgb(path: str, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def _read_depth(path: str, size: Tuple[int, int]) -> np.ndarray:
    if path.lower().endswith(".exr"):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif path.lower().endswith(".npy"):
        depth = np.load(path)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    return depth


def _read_mask(path: str, size: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return mask


def _norm_depth(depth: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    d = depth.copy().astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = (d - dmin) / max(dmax - dmin, 1e-6)  # [0,1]
    d = np.clip(d, 0.0, 1.0)
    d = d * 2.0 - 1.0  # [-1,1]
    return d


def _denorm_depth(dnorm: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    d = (dnorm + 1.0) / 2.0
    return d * (dmax - dmin) + dmin


def corrupt_depth_normalized(depth_norm: np.ndarray, mask: np.ndarray, sigma_mask: float, sigma_global: float) -> np.ndarray:
    h, w = depth_norm.shape
    global_noise = np.random.normal(0.0, sigma_global, size=(h, w)).astype(np.float32)
    mask_noise = np.random.normal(0.0, sigma_mask, size=(h, w)).astype(np.float32)
    raw = depth_norm + global_noise
    raw = raw + mask_noise * mask
    raw = np.clip(raw, -1.0, 1.0)
    return raw


class DepthInpaintDataset(Dataset):
    def __init__(self, cfg: DepthDataConfig, guidance_cfg: Optional[GuidanceConfig] = None):
        self.cfg = cfg
        self.gcfg = guidance_cfg or GuidanceConfig()

        # Check folders
        flat_rgb = os.path.join(cfg.root, cfg.rgb_dir)
        flat_depth = os.path.join(cfg.root, cfg.depth_dir)
        flat_mask = os.path.join(cfg.root, cfg.mask_dir)

        self.rgb_paths: List[str] = []
        self.depth_paths: List[str] = []
        self.mask_paths: List[str] = []
        self.guidance_paths: List[str] = []  # Pre-computed guidance maps

        # Determine which guidance folder to use based on branch
        guidance_dir_name = cfg.guidance_optical_dir if cfg.branch == "optical" else cfg.guidance_geometric_dir

        if os.path.isdir(flat_rgb) and os.path.isdir(flat_depth) and os.path.isdir(flat_mask):
            self.rgb_paths = self._collect(flat_rgb, IMG_EXTS)
            self.depth_paths = self._collect(flat_depth, DEPTH_EXTS)
            self.mask_paths = self._collect(flat_mask, MASK_EXTS)
            # Check for flat guidance folder
            flat_guidance = os.path.join(cfg.root, guidance_dir_name)
            if os.path.isdir(flat_guidance):
                self.guidance_paths = self._collect(flat_guidance, IMG_EXTS)
        else:
            # Fallback for nested structure (class/folder/...)
            if not os.path.isdir(cfg.root):
                raise RuntimeError(f"Root directory does not exist: {cfg.root}")

            class_folders = [os.path.join(cfg.root, d) for d in os.listdir(cfg.root)
                           if os.path.isdir(os.path.join(cfg.root, d))
                           and not d.startswith("guidance")]  # Skip guidance folders at root
            class_folders.sort()

            for cf in class_folders:
                rgb_dir = os.path.join(cf, cfg.rgb_dir)
                depth_dir = os.path.join(cf, cfg.depth_dir)
                mask_dir = os.path.join(cf, cfg.mask_dir)
                guidance_dir = os.path.join(cf, guidance_dir_name)

                if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir) and os.path.isdir(mask_dir)):
                    continue
                rgbs = self._collect(rgb_dir, IMG_EXTS)
                depths = self._collect(depth_dir, DEPTH_EXTS)
                masks = self._collect(mask_dir, MASK_EXTS)
                guides = self._collect(guidance_dir, IMG_EXTS) if os.path.isdir(guidance_dir) else []

                n_local = min(len(rgbs), len(depths), len(masks))
                if n_local == 0:
                    continue
                self.rgb_paths.extend(rgbs[:n_local])
                self.depth_paths.extend(depths[:n_local])
                self.mask_paths.extend(masks[:n_local])
                # Extend guidance paths, padding with empty strings if not available
                if len(guides) >= n_local:
                    self.guidance_paths.extend(guides[:n_local])
                else:
                    self.guidance_paths.extend(guides + [""] * (n_local - len(guides)))

        n = min(len(self.rgb_paths), len(self.depth_paths), len(self.mask_paths))
        self.rgb_paths = self.rgb_paths[:n]
        self.depth_paths = self.depth_paths[:n]
        self.mask_paths = self.mask_paths[:n]
        self.guidance_paths = self.guidance_paths[:n] if self.guidance_paths else [""] * n

        # Check if we have pre-computed guidance
        self.has_precomputed = len(self.guidance_paths) > 0 and self.guidance_paths[0] != ""
        if self.has_precomputed and cfg.use_precomputed_guidance:
            print(f"[INFO] Using pre-computed {cfg.branch} guidance maps from '{guidance_dir_name}/'")
        elif cfg.include_guidance:
            print(f"[INFO] Computing {cfg.branch} guidance maps on-the-fly (slower)")

        if n == 0:
            print(f"[WARNING] No data found in {cfg.root}. Check folder structure.")

    def _collect(self, folder: str, exts: Tuple[str, ...]) -> List[str]:
        if not os.path.isdir(folder):
            return []
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()
        return files

    def __len__(self):
        return len(self.rgb_paths)

    def _load_precomputed_guidance(self, idx: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load pre-computed guidance map if available."""
        if idx >= len(self.guidance_paths) or not self.guidance_paths[idx]:
            return None
        path = self.guidance_paths[idx]
        if not os.path.exists(path):
            return None
        guide = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if guide is None:
            return None
        guide = cv2.resize(guide, size, interpolation=cv2.INTER_NEAREST)
        guide = guide.astype(np.float32) / 255.0  # [0, 1]
        return guide

    def __getitem__(self, idx: int):
        c = self.cfg
        rgb = _read_rgb(self.rgb_paths[idx], c.size)
        depth = _read_depth(self.depth_paths[idx], c.size)
        mask = _read_mask(self.mask_paths[idx], c.size)  # 1=transparent area

        # normalize depth to [-1,1]
        depth_norm = _norm_depth(depth, *c.depth_minmax)

        # -------------------------------------------------------------------
        # DITR Preparation: Zero out depth inside masked (transparent) area
        # No noise/corruption - just simulate missing depth from sensor
        # -------------------------------------------------------------------
        # Raw depth = normalized depth with transparent region zeroed out
        raw_depth = depth_norm * (1.0 - mask)  # Zero inside mask (glass area)

        # -------------------------------------------------------------------
        # Branch-specific conditioning
        # -------------------------------------------------------------------
        # Try to load pre-computed guidance first
        guide = None
        if c.use_precomputed_guidance and self.has_precomputed:
            guide = self._load_precomputed_guidance(idx, c.size)

        if c.branch == "optical":
            # Track A: Optical Branch (Inpaints Glass Region)
            # Conditioning: RGB + raw_depth (zeroed in glass) + M_DA guidance
            # Mask: Used for inference merging only (not for loss weighting)
            # Both models train on FULL depth image - conditioning differentiates them
            loss_mask = mask  # Kept for inference merging
            cond_depth = raw_depth  # Already zeroed in glass area
            if guide is None and c.include_guidance:
                # Fallback: compute on-the-fly
                zeroed_depth = depth * (1.0 - mask)
                mda_full =  m_da(rgb, depth, self.gcfg, mask=None, zeroed_depth=zeroed_depth)

                # get background edges for outside masked area
                mrgb_full = m_rgb(rgb, self.gcfg, mask=None)

                # combine both maps: use m_da inside mask, m_rgb outside
                guide = (mask * mda_full) + ((1.0 - mask) * mrgb_full)

            elif guide is None:
                guide = np.zeros_like(mask)

        else:  # geometric
            # Track B: Geometric Branch (Refines Background)
            # Conditioning: RGB + raw_depth + M_RGB guidance
            # Mask: Used for inference merging only (not for loss weighting)
            # Both models train on FULL depth image - conditioning differentiates them
            loss_mask = 1.0 - mask  # Kept for inference merging
            cond_depth = raw_depth  # Same raw depth for geometric
            if guide is None and c.include_guidance:
                #trained on everywhere
                guide = m_rgb(rgb, self.gcfg, mask=None)
            elif guide is None:
                guide = np.zeros_like(mask)

        # to torch tensors
        rgb_t = torch.from_numpy(rgb[:, :, ::-1].copy()).float() / 255.0  # CHW
        rgb_t = rgb_t.permute(2, 0, 1)

        depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)  # GT
        cond_depth_t = torch.from_numpy(cond_depth).float().unsqueeze(0) # Masked Raw
        guide_t = torch.from_numpy(guide).float().unsqueeze(0)
        loss_mask_t = torch.from_numpy(loss_mask).float().unsqueeze(0)

        # Conditioning: [RGB (3), Masked_Raw_Depth (1), Guidance (1)]
        cond = torch.cat([rgb_t, cond_depth_t, guide_t], dim=0)

        batch = {
            "pixel_values": depth_t,        # target clean depth (FULL image)
            "conditioning": cond,           # RGB + Masked Raw + Guidance (branch-specific)
            "loss_mask": loss_mask_t,      # mask for inference merging (not used in loss)
        }
        return batch
