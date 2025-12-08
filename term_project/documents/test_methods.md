Based on the DITR paper, here is how the researchers tested their model on the ClearGrasp dataset and how you can replicate their metrics.

### 1\. The Test Set Protocol

[cite_start]The paper is explicit about how they handled the ClearGrasp testing[cite: 291, 296, 299]:

  * **Dataset Split:** They used the **Real-World** portion of ClearGrasp for testing (approx. 286 images), not the synthetic validation set.
  * **The "Ground Truth" Problem:** The real-world dataset was collected with a RealSense D435 camera, which has missing depth values (holes) in the ground truth itself (unlike the perfect synthetic data).
  * **Metric Masking (Crucial):** They **only calculate metrics on pixels where valid Ground Truth exists**.
    > [cite_start]*"For quantitative results comparisons, we only calculate the metrics on pixels with ground truth."* [cite: 299]
    > This means if a pixel has a `NaN` or `0` in the ground truth file, you must **ignore it** in your error calculation, even if your model generated a depth value there.

### 2\. The Metrics to Generate

The paper uses standard depth estimation metrics defined in Eq. [cite_start]4-6 and Section V-B[cite: 124, 132, 281]. You should implement these functions:

1.  **RMSE (Root Mean Square Error):** Measures standard error.
2.  **MAE (Mean Absolute Error):** Measures average magnitude of errors.
3.  **REL (Mean Relative Error):** Measures error relative to the distance (important because 1cm error at 10m is less bad than 1cm error at 0.5m).
4.  **$\delta$ (Threshold Accuracy):** The percentage of pixels where the ratio between prediction and ground truth is small ($max(\frac{pred}{gt}, \frac{gt}{pred}) < 1.25^n$). The paper uses $1.05$, $1.10$, and $1.25$.

### 3\. Python Implementation for Metrics

Since you saved your models as `.pt` files, you can write a script to load them, run inference, and calculate these metrics.

Here is a robust implementation of the metrics that handles the "Valid Ground Truth Only" rule:

```python
import numpy as np
import torch

def compute_errors(gt, pred):
    """
    Computes metrics for a single image or batch.

    Args:
        gt (np.ndarray or torch.Tensor): Ground Truth depth (meters).
        pred (np.ndarray or torch.Tensor): Predicted depth (meters).

    Returns:
        dict: Dictionary of metrics
    """
    # Ensure numpy
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()

    # 1. MASKING: Only evaluate valid GT pixels
    # We assume valid GT > 0.001 (to avoid div by zero) and not NaN
    mask = (gt > 0.001) & (gt < 10.0) & np.isfinite(gt)

    # If using the DITR logic, you might also want to mask out the
    # "non-glass" area if you only care about the Optical Branch performance,
    # but for the "Overall" table, they likely evaluated the whole valid image.

    # Apply mask
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    n_valid = len(gt_valid)
    if n_valid == 0:
        return None

    # 2. Calculate Metrics
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))

    # Delta Metrics (Accuracy thresholds)
    a1 = (thresh < 1.05).mean()
    a2 = (thresh < 1.10).mean()
    a3 = (thresh < 1.25).mean()

    # Standard Error Metrics
    rmse = (gt_valid - pred_valid) ** 2
    rmse = np.sqrt(rmse.mean())

    mae = np.abs(gt_valid - pred_valid).mean()

    # REL (Mean Relative Error)
    rel = np.abs(gt_valid - pred_valid) / gt_valid
    rel = rel.mean()

    return {
        "RMSE": rmse,
        "MAE": mae,
        "REL": rel,
        "Delta1.05": a1,
        "Delta1.10": a2,
        "Delta1.25": a3
    }

# --- Example Usage Logic ---
# 1. Load your model
# model = torch.load("your_model.pt")
# model.eval()

# 2. Loop through test set
# metrics_list = []

# for rgb, raw_depth, gt_depth, mask in test_loader:
    # Run Inference
    # with torch.no_grad():
        # pred_depth = model(rgb, raw_depth)

    # Resize prediction to Full Resolution if needed (to match GT)
    # pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

    # Calculate Metrics
    # m = compute_errors(gt_depth, pred_depth)
    # if m: metrics_list.append(m)

# 3. Average over dataset
# final_metrics = {k: np.mean([x[k] for x in metrics_list]) for k in metrics_list[0]}
# print(final_metrics)
```
