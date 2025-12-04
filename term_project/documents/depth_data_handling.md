Robustly handling depth data from `.exr` (OpenEXR) files requires navigating three main challenges: the **high dynamic range** (floating point values), the presence of **invalid data** (Infinite/NaN values), and **inconsistent channel naming conventions**.

Based on industry standards and the context of depth inpainting, here is a robust workflow for processing EXR depth data.

### 1\. The Robust Loading Strategy

Unlike standard images, EXR files can store arbitrary channels. You cannot simply assume "Red" is depth.

  * **Challenge:** The depth data might be in a channel named `Z`, `Depth.Z`, or simply encoded into `R` (Red), `G`, or `B`.
  * **Solution:** Inspect the header first to find the correct channel.
  * **The "Window" Trap:** EXR files have a `displayWindow` (total canvas) and a `dataWindow` (where pixels actually exist). If you simply read the bytes without accounting for the offset, your image will be shifted or cropped.

### 2\. Sanitizing the Data (Handling NaNs and Infs)

This is the most critical step for robustness. Raw depth maps frequently contain:

  * **$\infty$ (Infinity):** Represents the sky or areas beyond the sensor's range.
  * **`NaN` (Not a Number):** Represents sensor errors, stereo matching failures, or invalid calculations.

If you feed these directly into a neural network (like DITR) or a visualization tool, the math will break.

**Best Practice for Sanitization:**

1.  **Mask Validity:** Create a boolean mask of valid pixels.
2.  **Clip Distance:** Replace `inf` with a "Far Plane" value (e.g., 655m or 100m, depending on your scene scale).
3.  **Fill NaNs:** Replace `NaN` with 0 (if 0 = missing) or the Far Plane value, depending on your convention.

### 3\. Normalization Strategies

Raw depth is typically **linear distance** (in meters or blender units). For machine learning or visualization, linear depth is often poor because the difference between 1m and 2m is far more significant than 100m and 101m, yet the numerical difference is the same.

  * **Standard Linear:** $D_{norm} = \frac{D - min}{max - min}$ (Good for simple visualization, bad for ML).
  * **Inverse Depth (Disparity):** $D_{inv} = \frac{1}{D + \epsilon}$ (Best for ML/DITR). This emphasizes close objects and naturally compresses the "sky" (infinity) toward zero.

-----

### Comprehensive Python Implementation

This script handles the file rigorously using `OpenEXR` and `numpy`.

```python
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt

def read_robust_exr_depth(file_path, max_far_plane=100.0):
    """
    Robustly reads a depth map from an EXR file.

    Args:
        file_path: Path to the .exr file
        max_far_plane: The value to use for "Infinity" (the sky)

    Returns:
        depth_map: Sanitized numpy float32 array (Linear Depth)
        valid_mask: Boolean array where True means valid data (not NaN/Inf)
    """

    # 1. Open the file
    if not OpenEXR.isOpenExrFile(file_path):
        raise ValueError("File is not a valid EXR")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()

    # 2. robustly find the depth channel
    # Common names: 'Z', 'Depth.Z', 'R' (if saved as RGB), 'CamelCase.Z'
    channels = header['channels'].keys()
    depth_channel = None

    # Priority list for channel names
    priorities = ['Z', 'Depth.Z', 'depth.Z', 'depth.z', 'R']

    for candidate in priorities:
        if candidate in channels:
            depth_channel = candidate
            break

    if depth_channel is None:
        # Fallback: take the first channel available if strict naming fails
        depth_channel = list(channels)[0]
        print(f"Warning: ambiguous channels {channels}. Using '{depth_channel}' as depth.")

    # 3. Handle Data Window (Crucial for correct alignment)
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the bytes
    raw_bytes = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))

    # Convert to numpy
    depth_raw = np.frombuffer(raw_bytes, dtype=np.float32)
    depth_raw = depth_raw.reshape((height, width))

    # 4. Sanitization (The "Robust" Part)

    # Mask out NaNs (Errors) and Infs (Sky)
    # Note: np.isfinite returns True for regular numbers, False for NaN/Inf
    valid_mask = np.isfinite(depth_raw)

    # Make a copy to avoid modifying original if needed later
    depth_sanitized = depth_raw.copy()

    # Handle NaNs: Set to 0 (or max_far_plane, depending on preference)
    depth_sanitized = np.nan_to_num(depth_sanitized, nan=0.0, posinf=max_far_plane, neginf=0.0)

    # Clamp extreme values (just in case of outliers > max_far_plane)
    depth_sanitized = np.clip(depth_sanitized, 0, max_far_plane)

    return depth_sanitized, valid_mask

# --- Usage Example ---
file_path = "render_depth.exr"

try:
    depth, mask = read_robust_exr_depth(file_path)

    # 5. Inverse Depth Normalization (Great for Visualization/ML)
    # We add a small epsilon (1e-5) to avoid division by zero
    inverse_depth = 1.0 / (depth + 1e-5)

    # Visualize
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Linear Depth (Clipped)")
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label="Meters")

    plt.subplot(1, 2, 2)
    plt.title("Inverse Depth (1/Z)")
    plt.imshow(inverse_depth, cmap='magma')
    plt.colorbar(label="Proximity")

    plt.show()

except Exception as e:
    print(f"Error processing EXR: {e}")
```

### Key Takeaways for DITR Context

If you are preparing data for the DITR model mentioned in your PDF:

  * **The "Holes" are Features:** DITR explicitly looks for discrepancies. You should likely treat `NaN` or `0` values as your **"Missing Depth"** mask.
  * **Inverse Depth:** Most diffusion models for depth (like the ones cited in the paper) operate in **inverse depth** space or latent space to maintain detail in close-up objects (where transparent items usually are).
  * **Optical vs Geometric:** When you sanitize, remember that `NaNs` in the raw sensor data often correspond to the "Optical Depth Loss" the paper discusses (glass/mirrors). Don't just overwrite them with the background; note their location—that is where your inpainting mask comes from.