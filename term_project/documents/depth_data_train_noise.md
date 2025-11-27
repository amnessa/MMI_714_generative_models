1. Where to Change the Corruption Code

You need to modify the __getitem__ method in your ClearGraspDataset class (likely inside your notebook or a dataset.py file).

Currently, it looks like this:
Python

# Current Logic: Hard Zeroing
input_depth_tensor = target_depth_tensor.clone()
input_depth_tensor[mask_tensor > 0.0] = -1.0 # Set to "empty"

2. How to Implement "Realistic Noise"

You want to simulate the chaotic scattering of IR light on transparent surfaces. This isn't just uniform Gaussian noise; it's often a mix of:

    Dropouts (Zeros): The sensor gets no return. (You already have this).

    Outliers (Spikes): The sensor gets a reflection from a wall behind the object or a weird angle, resulting in random depth values.

    Gaussian Noise: General sensor fuzz.

Here is a robust implementation plan for your __getitem__:
Python

# Inside ClearGraspDataset class

def apply_realistic_corruption(self, depth, mask):
    """
    depth: (1, H, W) normalized tensor [-1, 1]
    mask: (1, H, W) normalized tensor [-1, 1] (where > 0 is object)
    """
    corrupted = depth.clone()

    # 1. Get indices of the transparent object
    object_indices = (mask > 0.0)

    # 2. Probability of Dropout (Zeroing) - e.g., 60% of pixels are just lost
    dropout_prob = 0.6
    dropout_mask = (torch.rand_like(depth) < dropout_prob) & object_indices
    corrupted[dropout_mask] = -1.0 # Set to "empty"

    # 3. Probability of Noise (The remaining 40% get noisy)
    # We only add noise to object pixels that were NOT dropped out
    noise_indices = object_indices & (~dropout_mask)

    if noise_indices.any():
        # Add strong Gaussian noise to simulate scattering
        # Scale of noise: 0.5 is huge in [-1, 1] space, simulating wild reflections
        noise = torch.randn_like(depth) * 0.5
        corrupted[noise_indices] += noise[noise_indices]

        # Clip to ensure we stay in valid range [-1, 1]
        corrupted = torch.clamp(corrupted, -1.0, 1.0)

    return corrupted

# In __getitem__:

if self.split == "synthetic":
    # ... load depth and mask ...
    input_depth_tensor = self.apply_realistic_corruption(target_depth_tensor, mask_tensor)

3. Why this is better (and safer)

    Realism: Real sensors on glass bottles often show a "speckled" pattern—some black holes, some random bright spots. This simulates that texture.

    Robustness: By randomizing the noise every time (torch.randn), you are effectively doing Data Augmentation. The model never sees the exact same "broken bottle" twice. It learns to be robust to any kind of sensor failure.

4. Comparison to VAE/Reconstruction

Your intuition about VAEs is spot on.

    VAE: "I will map this noisy input to a latent vector z, and then decode z to a clean output."

    Your Diffusion Model: "I will take this noisy input as a hint. I will start with pure static (random noise), and I will slowly sculpt it into a clean depth map, using the hint to tell me where the edges are."

By changing the corruption to be "noisy" instead of just "empty," you make the "hint" slightly more confusing, which forces the model to rely even more heavily on the RGB image. This is exactly what you want.