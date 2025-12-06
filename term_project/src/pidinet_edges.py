"""
PiDiNet Edge Detector Wrapper for DITR Pipeline

This module provides a clean interface to PiDiNet (Pixel Difference Networks)
for edge detection on both RGB images and depth maps.

Paper: "Pixel Difference Networks for Efficient Edge Detection" (ICCV 2021)
GitHub: https://github.com/hellozhuo/pidinet
"""

import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ============================================================================
# PiDiNet Architecture (Converted/Inference-only version)
# ============================================================================

class CSAM(nn.Module):
    """Compact Spatial Attention Module"""
    def __init__(self, channels: int):
        super().__init__()
        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class CDCM(nn.Module):
    """Compact Dilation Convolution based Module"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(x)
        x = self.conv1(x)
        return self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)


class MapReduce(nn.Module):
    """Reduce feature maps into a single edge map"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PDCBlock(nn.Module):
    """PDC Block (converted to vanilla CNN for inference)"""
    def __init__(self, pdc_type: str, inplane: int, outplane: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, outplane, kernel_size=1, padding=0)

        # For converted model: rd uses 5x5, others use 3x3
        if pdc_type == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)

        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, outplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class PiDiNet(nn.Module):
    """
    PiDiNet architecture (converted version for inference).

    Default config: 'carv4' with sa=True, dil=True (table5_pidinet)
    """
    def __init__(
        self,
        inplane: int = 60,
        pdcs: Optional[List[str]] = None,
        dil: Optional[int] = 24,
        sa: bool = True,
    ):
        super().__init__()
        self.sa = sa
        self.dil = dil

        # Default carv4 config
        if pdcs is None:
            pdcs = ['cd', 'ad', 'rd', 'cv', 'cd', 'ad', 'rd', 'cv',
                    'cd', 'ad', 'rd', 'cv', 'cd', 'ad', 'rd', 'cv']

        self.fuseplanes = []
        self.inplane = inplane

        # Initial block
        init_kernel = 5 if pdcs[0] == 'rd' else 3
        init_pad = 2 if pdcs[0] == 'rd' else 1
        self.init_block = nn.Conv2d(3, self.inplane, kernel_size=init_kernel, padding=init_pad, bias=False)

        # Block 1
        self.block1_1 = PDCBlock(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = PDCBlock(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = PDCBlock(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        # Block 2
        inplane_prev = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = PDCBlock(pdcs[4], inplane_prev, self.inplane, stride=2)
        self.block2_2 = PDCBlock(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = PDCBlock(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = PDCBlock(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        # Block 3
        inplane_prev = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = PDCBlock(pdcs[8], inplane_prev, self.inplane, stride=2)
        self.block3_2 = PDCBlock(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = PDCBlock(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = PDCBlock(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        # Block 4
        self.block4_1 = PDCBlock(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = PDCBlock(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = PDCBlock(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = PDCBlock(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        # Fusion modules
        self.conv_reduces = nn.ModuleList()
        if sa and dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], dil))
                self.attentions.append(CSAM(dil))
                self.conv_reduces.append(MapReduce(dil))
        elif sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], dil))
                self.conv_reduces.append(MapReduce(dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        # Final classifier
        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the fused edge map.

        Args:
            x: (B, 3, H, W) RGB image tensor in [0, 1] range

        Returns:
            (B, 1, H, W) edge probability map in [0, 1] range
        """
        H, W = x.shape[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        features = [x1, x2, x3, x4]

        if self.sa and self.dil is not None:
            for i, xi in enumerate(features):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate(features):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate(features):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = features

        # Multi-scale edge maps
        e1 = F.interpolate(self.conv_reduces[0](x_fuses[0]), (H, W), mode="bilinear", align_corners=False)
        e2 = F.interpolate(self.conv_reduces[1](x_fuses[1]), (H, W), mode="bilinear", align_corners=False)
        e3 = F.interpolate(self.conv_reduces[2](x_fuses[2]), (H, W), mode="bilinear", align_corners=False)
        e4 = F.interpolate(self.conv_reduces[3](x_fuses[3]), (H, W), mode="bilinear", align_corners=False)

        # Fused output
        output = self.classifier(torch.cat([e1, e2, e3, e4], dim=1))
        return torch.sigmoid(output)


# ============================================================================
# Weight Conversion Utilities (for loading original PiDiNet checkpoints)
# ============================================================================

def convert_pdc_weight(op: str, weight: torch.Tensor) -> torch.Tensor:
    """Convert PDC convolution weights to vanilla CNN weights."""
    if op == 'cv':
        return weight
    elif op == 'cd':  # Central difference
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        return weight.view(shape)
    elif op == 'ad':  # Angular difference
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv
    elif op == 'rd':  # Radial difference (3x3 -> 5x5)
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 25, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        return buffer.view(shape[0], shape[1], 5, 5)
    else:
        raise ValueError(f"Unknown PDC type: {op}")


def convert_pidinet_checkpoint(state_dict: dict, config: str = 'carv4') -> dict:
    """
    Convert original PiDiNet checkpoint to converted (vanilla CNN) format.

    Args:
        state_dict: Original checkpoint state dict
        config: PDC configuration name (default: 'carv4')

    Returns:
        Converted state dict for PiDiNet in inference mode
    """
    # carv4 config
    pdcs = ['cd', 'ad', 'rd', 'cv', 'cd', 'ad', 'rd', 'cv',
            'cd', 'ad', 'rd', 'cv', 'cd', 'ad', 'rd', 'cv']

    new_dict = {}
    pdc_layers = [
        ('init_block.weight', 0),
        ('block1_1.conv1.weight', 1),
        ('block1_2.conv1.weight', 2),
        ('block1_3.conv1.weight', 3),
        ('block2_1.conv1.weight', 4),
        ('block2_2.conv1.weight', 5),
        ('block2_3.conv1.weight', 6),
        ('block2_4.conv1.weight', 7),
        ('block3_1.conv1.weight', 8),
        ('block3_2.conv1.weight', 9),
        ('block3_3.conv1.weight', 10),
        ('block3_4.conv1.weight', 11),
        ('block4_1.conv1.weight', 12),
        ('block4_2.conv1.weight', 13),
        ('block4_3.conv1.weight', 14),
        ('block4_4.conv1.weight', 15),
    ]

    for pname, p in state_dict.items():
        converted = False
        for layer_name, pdc_idx in pdc_layers:
            if layer_name in pname:
                new_dict[pname] = convert_pdc_weight(pdcs[pdc_idx], p)
                converted = True
                break
        if not converted:
            new_dict[pname] = p

    return new_dict


# ============================================================================
# PiDiNet Edge Detector Wrapper
# ============================================================================

@dataclass
class PiDiNetConfig:
    """Configuration for PiDiNet edge detector."""
    # Model settings
    model_path: Optional[str] = None  # Path to pretrained weights
    device: str = "cuda"
    dilation: int = 24      # CDCM dilation channels (must match pretrained weights = 24)

    # Edge detection settings
    threshold: float = 0.5  # Binarization threshold (0-1)
    use_nms: bool = False   # Apply non-maximum suppression (slower but cleaner)
    morph_kernel: int = 0   # Morphological closing kernel size (0=disabled)
    morph_erode: int = 1    # Erosion iterations to thin edges (0=disabled)

    # Depth-to-RGB conversion for depth edge detection
    depth_max_distance: float = 10.0
    depth_use_inverse: bool = True


# Global model cache
_PIDINET_CACHE = {}


def get_pidinet_model(cfg: PiDiNetConfig) -> PiDiNet:
    """Load and cache PiDiNet model."""
    global _PIDINET_CACHE

    cache_key = (cfg.model_path, cfg.device, cfg.dilation)
    if cache_key not in _PIDINET_CACHE:
        print(f"Loading PiDiNet model (dilation={cfg.dilation})...")
        model = PiDiNet(inplane=60, dil=cfg.dilation, sa=True)

        if cfg.model_path is not None and os.path.exists(cfg.model_path):
            print(f"  Loading weights from: {cfg.model_path}")
            checkpoint = torch.load(cfg.model_path, map_location='cpu', weights_only=True)

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel training)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                cleaned_state_dict[new_key] = v
            state_dict = cleaned_state_dict

            # ALWAYS convert official PiDiNet checkpoints - they use PDC format
            # The conversion transforms PDC conv weights to vanilla CNN weights
            print("  Converting PDC weights to vanilla CNN format...")
            state_dict = convert_pidinet_checkpoint(state_dict)

            # Load with strict=False to handle any missing keys
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Warning: Missing keys: {missing}")
            if unexpected:
                print(f"  Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Warning: Unexpected keys: {unexpected}")
            print("  Weights loaded successfully.")
        else:
            print("  WARNING: No pretrained weights - using random initialization!")

        model = model.to(cfg.device)
        model.eval()
        _PIDINET_CACHE[cache_key] = model
        print("PiDiNet model ready.")

    return _PIDINET_CACHE[cache_key]

    return _PIDINET_CACHE[cache_key]


def _morph_close(edges: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological closing to connect nearby edges."""
    if kernel_size <= 0:
        return edges
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def pidinet_edges_rgb(
    rgb_bgr: np.ndarray,
    cfg: PiDiNetConfig,
) -> np.ndarray:
    """
    Extract edges from RGB image using PiDiNet.

    Args:
        rgb_bgr: (H, W, 3) BGR image (OpenCV format), uint8 or float
        cfg: PiDiNetConfig

    Returns:
        (H, W) binary edge map in [0, 1] range
    """
    model = get_pidinet_model(cfg)

    # Prepare input
    if rgb_bgr.dtype == np.uint8:
        rgb = rgb_bgr.astype(np.float32) / 255.0
    else:
        rgb = rgb_bgr.astype(np.float32)

    # BGR -> RGB
    rgb = rgb[:, :, ::-1].copy()

    # To tensor: (H, W, 3) -> (1, 3, H, W)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    x = x.to(cfg.device)

    with torch.no_grad():
        edge_prob = model(x)  # (1, 1, H, W)

    # To numpy
    edge_map = edge_prob[0, 0].cpu().numpy()

    # Threshold to binary
    edges = (edge_map > cfg.threshold).astype(np.float32)

    # Optional morphological closing
    edges = _morph_close((edges * 255).astype(np.uint8), cfg.morph_kernel)

    # Optional morphological erosion to thin edges
    if cfg.morph_erode > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=cfg.morph_erode)

    edges = (edges > 0).astype(np.float32)

    return edges


def pidinet_edges_depth(
    depth: np.ndarray,
    cfg: PiDiNetConfig,
) -> np.ndarray:
    """
    Extract edges from depth map using PiDiNet.

    Converts depth to a 3-channel pseudo-RGB representation first.
    Uses inverse depth (disparity) for better near-object edge detection.

    Args:
        depth: (H, W) depth map (meters)
        cfg: PiDiNetConfig

    Returns:
        (H, W) binary edge map in [0, 1] range
    """
    model = get_pidinet_model(cfg)

    # Robust depth preprocessing
    d = depth.astype(np.float32)
    d = np.nan_to_num(d, nan=cfg.depth_max_distance, posinf=cfg.depth_max_distance, neginf=0.0)
    d = np.clip(d, 0.001, cfg.depth_max_distance)  # Avoid division by zero

    # Convert to inverse depth (disparity) for better edges
    if cfg.depth_use_inverse:
        # Use inverse depth but normalize to actual data range
        d_inv = 1.0 / d
        # Normalize to [0, 1] using min-max of the actual data
        d_min, d_max = d_inv.min(), d_inv.max()
        if d_max - d_min > 1e-6:
            d_norm = (d_inv - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d_inv)
    else:
        # Linear depth normalization
        d_norm = d / cfg.depth_max_distance

    d_norm = np.clip(d_norm, 0, 1)

    # Create 3-channel pseudo-RGB
    depth_rgb = np.stack([d_norm, d_norm, d_norm], axis=-1)

    # To tensor
    x = torch.from_numpy(depth_rgb).permute(2, 0, 1).unsqueeze(0).float()
    x = x.to(cfg.device)

    with torch.no_grad():
        edge_prob = model(x)

    edge_map = edge_prob[0, 0].cpu().numpy()

    # Threshold to binary
    edges = (edge_map > cfg.threshold).astype(np.float32)

    # Optional morphological closing
    edges = _morph_close((edges * 255).astype(np.uint8), cfg.morph_kernel)

    # Optional morphological erosion to thin edges
    if cfg.morph_erode > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=cfg.morph_erode)

    edges = (edges > 0).astype(np.float32)

    return edges


def compute_mda_pidinet(
    rgb_bgr: np.ndarray,
    depth: np.ndarray,
    cfg: PiDiNetConfig,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute M_DA (depth-aware guidance map) using PiDiNet edges.

    M_DA = M_RGB ∩ M_D (intersection of RGB and depth edges)

    This finds edges that appear in BOTH modalities - the "confirmed" boundaries.
    Edges only in RGB (not in M_DA) are likely transparent/reflective artifacts.

    Args:
        rgb_bgr: (H, W, 3) BGR image
        depth: (H, W) depth map
        cfg: PiDiNetConfig
        mask: Optional (H, W) mask where 1=transparent area

    Returns:
        (H, W) guidance map
    """
    e_rgb = pidinet_edges_rgb(rgb_bgr, cfg)
    e_depth = pidinet_edges_depth(depth, cfg)

    # Intersection: edges that appear in both modalities
    mda = np.minimum(e_rgb, e_depth)

    # Mask to transparent region only (for optical branch)
    if mask is not None:
        mda = mda * mask

    return mda.astype(np.float32)


def compute_mrgb_pidinet(
    rgb_bgr: np.ndarray,
    cfg: PiDiNetConfig,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute M_RGB guidance map using PiDiNet edges.

    Args:
        rgb_bgr: (H, W, 3) BGR image
        cfg: PiDiNetConfig
        mask: Optional (H, W) mask where 1=transparent area

    Returns:
        (H, W) guidance map (masked to background if mask provided)
    """
    edges = pidinet_edges_rgb(rgb_bgr, cfg)

    # Mask to background region only (for geometric branch)
    if mask is not None:
        edges = edges * (1.0 - mask)

    return edges.astype(np.float32)


# ============================================================================
# Download pretrained weights utility
# ============================================================================

def download_pidinet_weights(save_dir: str = ".") -> str:
    """
    Download pretrained PiDiNet weights from GitHub releases.

    Returns path to downloaded checkpoint.
    """
    import urllib.request

    # table5_pidinet checkpoint URL (from official repo)
    url = "https://github.com/hellozhuo/pidinet/raw/master/trained_models/table5_pidinet.pth"
    save_path = os.path.join(save_dir, "table5_pidinet.pth")

    if os.path.exists(save_path):
        print(f"Checkpoint already exists: {save_path}")
        return save_path

    print(f"Downloading PiDiNet weights from: {url}")
    os.makedirs(save_dir, exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")

    return save_path


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("PiDiNet Edge Detector Test")
    print("=" * 50)

    # Create a test image
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    test_depth = np.random.rand(128, 128).astype(np.float32) * 5.0

    # Initialize config (no pretrained weights for quick test)
    cfg = PiDiNetConfig(
        model_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=0.5,
    )

    print(f"Device: {cfg.device}")
    print(f"Testing with random {test_img.shape} image...")

    # Test RGB edges
    e_rgb = pidinet_edges_rgb(test_img, cfg)
    print(f"RGB edges shape: {e_rgb.shape}, range: [{e_rgb.min():.2f}, {e_rgb.max():.2f}]")

    # Test depth edges
    e_depth = pidinet_edges_depth(test_depth, cfg)
    print(f"Depth edges shape: {e_depth.shape}, range: [{e_depth.min():.2f}, {e_depth.max():.2f}]")

    # Test M_DA
    mda = compute_mda_pidinet(test_img, test_depth, cfg)
    print(f"M_DA shape: {mda.shape}, range: [{mda.min():.2f}, {mda.max():.2f}]")

    print("\n✓ PiDiNet module working correctly!")
