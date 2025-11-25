# Training Loop & Data Pipeline
### Next Steps: The Training Loop & Data Pipeline

Now that you have the **Model** (`ConditionalUNetDepth`), you need to build the **Diffusion Process** (the noise scheduler) and the **Data Pipeline** (the corruption strategy we discussed).

Here is the plan for the next phase of implementation, broken down into actionable steps.

#### 1\. Implement the Diffusion Logic (GaussianDiffusion)

You need a class to handle the forward process (adding noise) and the sampling process (reverse diffusion). Since you want to keep it self-contained, you can implement a simplified `GaussianDiffusion` class.

**Key Responsibilities of this Class:**

  * **`__init__`:** Pre-compute the noise schedule ($\beta_t, \alpha_t, \bar{\alpha}_t$). A standard linear schedule (e.g., $\beta_{start}=0.0001, \beta_{end}=0.02$) is fine for this.
  * **`q_sample` (Forward Process):** Takes ground truth $x_0$ and timestep $t$, returns noisy $x_t$ and the noise $\epsilon$.
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
  * **`p_losses` (Training Step):**
    1.  Sample random $t$ for a batch.
    2.  Generate random noise $\epsilon$.
    3.  Create noisy depth $x_t$ using `q_sample`.
    4.  **Crucial:** Create the "Input" for the model.
          * *Model Input:* Concatenate `x_t` (1ch) + `RGB` (3ch) + `Corrupted Depth` (1ch).
    5.  Pass this to your `ConditionalUNetDepth`.
    6.  Calculate MSE loss between *predicted noise* and *actual noise* $\epsilon$.
  * **`sample` (Inference/Generation):**
    1.  Start with random noise $x_T$.
    2.  Loop backwards from $T \to 0$.
    3.  At each step, feed the current $x_t$ and the **conditioning** (RGB + Corrupted Depth) to the model.
    4.  Use the predicted noise to calculate $x_{t-1}$ (using the standard DDPM update rule).

#### 2\. Implement the Dataset & Synthetic Corruption

You need a `torch.utils.data.Dataset` that loads your data and applies the corruption on the fly.

**Logic:**

```python
class ClearGraspDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        # ... list all files ...
        # ... store transform (resize to 256x256, normalize to [-1, 1]) ...

    def __getitem__(self, idx):
        # Load RGB, Perfect Depth, Mask
        rgb = ...
        gt_depth = ...
        mask = ...

        # Synthetic Corruption:
        # Wherever mask > 0 (transparent), set depth to 0 (or -1 if normalized)
        corrupted_depth = gt_depth.clone()
        corrupted_depth[mask > 0] = -1.0  # Assuming -1 is your "empty/far" value after norm

        # Normalization:
        # Map depth [0, max_dist] -> [-1, 1]
        # Map RGB [0, 255] -> [-1, 1]

        return {
            "pixel_values": gt_depth,      # The target (x0)
            "conditioning": torch.cat([rgb, corrupted_depth], dim=0), # The condition
            "mask": mask # Optional, for visualization
        }
```

#### 3\. The Training Loop

Finally, create a `train.py` script that brings them together.

1.  Instantiate `ConditionalUNetDepth`.
2.  Instantiate your `GaussianDiffusion` wrapper.
3.  Create `DataLoader`.
4.  Loop through epochs:
      * Get batch.
      * Run `diffusion.p_losses(batch)`.
      * Optimizer step.
      * **Logging:** Every 500 steps, save a sample image. Visualizing the progress is critical\!

### Immediate Action Item

I recommend creating a file named **`diffusion.py`** next. Copying the `GaussianDiffusion` logic from `lucidrains` is safe, but strip out the complex "objective" types (keep it simple: predict 'epsilon' noise) to keep your codebase understandable and debuggable.
