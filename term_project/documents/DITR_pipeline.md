Based on the paper "Diffusion-Based Depth Inpainting for Transparent and Reflective Objects" (DITR), here is the extraction of the Preliminary, Methodology, and Experiments sections. I have structured this to serve as an implementation guide for your project, focusing on the mathematical formulations and architectural details.

### I. Preliminary
This section defines the problem scope and the mathematical decomposition of depth loss.

* **Problem Formulation:** The goal is to deduce the real depth value matrix $\tilde{D}$ ($H \times W$) from a distorted input depth image $D_{in}$ and a corresponding RGB image $X$ .
    * The mapping is defined as $D_{out} = DITR(D_{in}|X)$ .
* **Decomposition of Depth Loss:** The authors observe that depth loss arises from two distinct causes:
    1.  **Optical Depth Loss:** Caused by transparent or reflective surfaces where the IR spectrum penetrates or reflects .
    2.  **Geometric Depth Loss:** Caused by occlusion and optical parallax between RGB and depth sensors .
* **Set Definition:** The input is divided into orthogonal parts based on these losses:
    $$X = X^{op} \cup X^{geo}, \quad D_{in} = D_{in}^{op} \cup D_{in}^{geo}$$
    where "op" denotes optical loss regions and "geo" denotes geometric loss regions . These regions are assumed to be disjoint: $X^{op} \cap X^{geo} = \{\phi\}$ .
* **Evaluation Metrics:** The objective is to minimize the loss between $D_{out}$ and $\tilde{D}$ using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Mean Relative Error (REL)  .

### II. Methodology
The proposed DITR framework is a two-stage network: **Region Proposal** (Stage 1) and **Depth Inpainting** (Stage 2) .

#### A. Stage One: Region Proposal
This stage acts as a classifier to segment transparent and reflective objects .
* **Architecture:** Uses **TROSNet** (we will use ready to use masks), an encoder-decoder network designed for transparent/reflective object segmentation .
* **Input/Output:** Takes RGB and Depth images; outputs a segmentation mask identifying areas of *optical* depth loss. The remaining area is attributed to *geometric* depth loss.
* **Region Refinement (Crucial Step):** To handle misclassified pixels that harm inpainting, post-processing is applied to the mask:
    1.  Median filter with a $7\times7$ kernel.
    2.  Dilation with a $5\times5$ kernel for 3 iterations.

#### B. Stage Two: Depth Inpainting
This stage uses a **Latent Diffusion Model (LDM)** approach to regenerate depth values, processing optical and geometric regions via separate branches.


**1. Latent Space Transformation:**
* To reduce complexity, pixels are transformed to latent space $L$ using a VQ-GAN-like structure .
* $L = Enc(D_{in})$ and $D_{out} = Dec(L)$.
* The sampling scale is set to 4 (requires 2 downsampling/upsampling operations) .

**2. Diffusion Block:**
* **Diffusion Process:** Simulates a Markov chain adding Gaussian noise:
    $$p_{\theta}(L_{t}|L_{0}) = \mathcal{N}(0, I)$$
    where $t$ is the time step.
* **Objective Function:** The model optimizes the noise prediction:
    $$L = \mathbb{E}_{Enc(x), L, \mathcal{N}(0,I)} || \epsilon - \epsilon_{\theta}(L_t, t, F(x)) ||_2^2$$
    where $F(x)$ is the feature map from the Feature Extractor.
* **Denoising U-Net:**     * Consists of 4 layers.
    * **Downsampling:** Consecutive $3\times3$ convolutions $\rightarrow$ ReLU $\rightarrow$ $2\times2$ Max Pooling (stride 2). Feature channels double at each layer.
    * **Upsampling:** $2\times2$ convolution kernel followed by two $3\times3$ convolutions and ReLU.
    * **Attention:** Uses self-attention and cross-attention where latent representation $L_t$ acts as Value and Key.

**3. Feature Extractors (Guidance):**
Two separate extractors are used for the different loss types :
* **Geometric Inpainting:** Uses the RGB boundary map ($M_{RGB}$) directly.
* **Optical Inpainting:** Uses a **Depth-Aware boundary detection Map ($M_{DA}$)** .
    * Implementation uses **ViT-L SAM** (Segment Anything Model) on both RGB and Depth images to get boundary maps $M_{RGB}$ and $M_{D}$ .
    * Formula: $M_{DA} = M_{RGB} \setminus \mathcal{C}_{U}M_{D}$ (The set difference between RGB boundaries and the complement of depth boundaries) .

### III. Experiments
This section details the setup used to validate the implementation.

* **Datasets Used:**
    * **ClearGrasp:** 286 real-world RGB-D sets (Transparent objects) .
    * **TODD:** 57,715 samples (Transparent objects like beakers) .
    * **STD:** 27,000 samples (Transparent and Reflective/Specular objects) .
* **Training Implementation Details:**
    * **Stage 1 (Region Proposal):** SGD optimizer, Batch size 16, Momentum 0.9, Weight decay 0.0001 . Training epochs: 200; Fine-tuning epochs: 50 (LR 1e-3) .
    * **Stage 2 (Diffusion):** Diffusion steps $T=1,000$, Batch size 64, Learning rate 1e-6 .
    * **Hardware:** Trained on NVIDIA RTX 3090. (But I have rtx 4060 8gb vram)
* **Quantitative Results (RMSE):**
    * **ClearGrasp:** DITR achieved **0.019**, outperforming DFNet (0.025) and TranspareNet (0.026) .
    * **TODD:** DITR achieved **0.017**, surpassing SwinDRNet (0.033).
    * **STD:** DITR achieved **0.009** .
* **Inference Speed:** The inference latency is approximately 2.2 seconds (Stage 1: 43ms + Stage 2: 2256ms) on $1280 \times 720$ images.

**Next Step:**
Would you like me to outline a PyTorch file structure for the **Region Proposal** stage first, or would you prefer to start implementing the **Diffusion Block** and Loss function?