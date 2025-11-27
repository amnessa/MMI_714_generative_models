
2

Automatic Zoom
Project Proposal: Generative Depth
Completion for Transparent Objects
Student Name: Çağdaş Güven
Course: MMI714 – Generative Models for Multimedia
Date: 23 November 2025
1. Problem Definition
Accurate depth perception is a cornerstone of autonomous robotic manipulation, enabling
tasks such as grasping, path planning, and collision avoidance. However, standard RGB-D
sensors (e.g., RealSense, Kinect) fail catastrophically when encountering transparent or highly
reflective objects—such as plastic bottles, glassware, or polished metal parts common in
industrial manufacturing cells. Due to the physics of light refraction and specular reflection,
these sensors produce depth maps riddled with "holes," noise, and varying artifacts,
rendering the data unusable for precise interaction.
The scientific problem is that the missing depth information cannot be recovered by simple
deterministic interpolation or filtering, as the geometry is lost. This project frames the problem
as a conditional generation task: given the RGB context (which clearly shows the object)
and the incomplete depth map, can we generate a statistically plausible and geometrically
accurate completion of the depth map?
2. Solution and Hypothesis
I propose a Pixel-Space Conditional Denoising Diffusion Probabilistic Model (cDDPM) to
perform depth inpainting. Unlike discriminative regression models (e.g., standard U-Nets
trained with MSE loss) which tend to produce blurry outputs in uncertain regions, a diffusion
model learns the explicit data distribution of valid depth geometries.
Hypothesis: A generative diffusion model conditioned on both the RGB image and the raw
(noisy) depth sensor data will outperform standard interpolation techniques and
regression-based baselines in recovering the geometry of transparent objects. By learning
the joint distribution of visual appearance (RGB) and geometry (Depth), the model can
"hallucinate" the correct curvature and fill in the sensor holes consistent with the object's
visual features.
3. Experiment Plan
The project will be executed in three phases, optimized for our term timeline and limited
computational resources (RTX 4060 Mobile, 8GB VRAM):
1.  Data Preparation:
○  I will curate a lightweight training subset from the ClearGrasp dataset (approx.
5k-10k images).
○  Images will be resized to   or    to ensure the model fits within
GPU memory during training.
○  Depth values will be normalized to the range   to match the standard
diffusion process.
2.  Model Implementation:
○  I will base my code on the lucidrains/denoising-diffusion-pytorch repository,
extending the standard U-Net architecture.
○  Architecture: A Conditional U-Net operating in pixel space. The input will be
modified to accept a 4-channel conditioning tensor (3 channels for RGB + 1
channel for the raw/incomplete depth) concatenated with the noisy target.
○  The network will predict the noise residual added to the "clean" ground truth
depth.
3.  Evaluation & Future Work:
○  Quantitative: I will calculate Root Mean Square Error (RMSE) and Structural
Similarity Index (SSIM) on the test set, specifically masking the evaluation to the
transparent object regions.
○  Qualitative: I will visually compare the generated depth maps against the raw
sensor input and a zero-shot baseline using the pre-trained Marigold model
(running in inference-only mode).
○  Future Work: While outside the scope of this term project, the ultimate goal is to
integrate this trained model into my Master's thesis digital twin to enable robust
grasping of transparent parts in a jigless manufacturing cell.
4. Dataset
I will use the ClearGrasp dataset. It is uniquely suited for this research because it provides a
perfect paired ground truth for real-world transparent objects. The authors created this
dataset by capturing a scene with transparent objects and then physically swapping them
with identical spray-painted (opaque) counterparts to capture the "true" depth.
●  Input: RGB Image + Raw Depth (from RealSense D435, containing holes).
●  Target: Ground Truth Depth (from the opaque swap).
This dataset allows for fully supervised training of the generative model while addressing a
real-world domain gap in robotics.
