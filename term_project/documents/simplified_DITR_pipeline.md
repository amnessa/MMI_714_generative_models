Here is a simplified pipeline explaining exactly where the data goes and how the conditions are created.

### The DITR Pipeline: A Step-by-Step Flow

The big idea is that the system treats **glass/mirrors** (Optical Loss) and **background holes** (Geometric Loss) as two separate problems. It splits the image, fixes each part separately, and then glues them back together.

---

### Step 1: The Input & The Split
Everything starts with your two raw images from the robot's camera.
* **Input 1:** RGB Image (Color)
* **Input 2:** Raw Depth Image (Distance)

These inputs immediately go to **Stage 1** to answer the question: *"Where is the glass?"*

### Step 2: Generating the "Map" (Region Proposal)
This stage builds the mask that routes traffic for the rest of the pipeline.
* [cite_start]**Action:** Feed **RGB + Raw Depth** into the **TROSNet** neural network[cite: 178].
* **Output:** A binary **Mask**.
    * **White (1):** "This area is Glass/Mirror."
    * **Black (0):** "This area is Background."
* [cite_start]**Refinement:** The raw mask is often messy, so it goes through a "cleanup" filter (Median Filter + Dilation) to smooth the edges[cite: 215].

---

### Step 3: Branching (Where the Data Goes)
Now the pipeline splits into two parallel tracks. The **Mask** acts as the traffic cop.

#### **Track A: The Optical Branch (Fixing Glass)**
This branch only cares about the pixels inside the glass/mirror region.
1.  **Depth Input:** Take **Raw Depth** $\times$ **Mask**. (This zeroes out the background, leaving only glass depth) [cite_start][cite: 169].
2.  **Condition/Guidance ($M_{DA}$):** How does the model know shape of the glass?
    * It compares RGB edges vs. Depth edges.
    * **Logic:** "If I see an edge in the Color photo, but *no* edge in the Depth map, that is a ghost edge caused by transparency."
    * [cite_start]**Formula:** $M_{DA} = RGB\_Edges - Depth\_Edges$[cite: 258].
    * [cite_start]This $M_{DA}$ map is fed into the diffusion model to tell it: *"Draw depth boundaries here."* [cite: 261]

#### **Track B: The Geometric Branch (Fixing Background)**
This branch only cares about the pixels *outside* the glass.
1.  **Depth Input:** Take **Raw Depth** $\times$ **(1 - Mask)**. (This zeroes out the glass, leaving only background depth) [cite_start][cite: 179].
2.  **Condition/Guidance ($M_{RGB}$):**
    * It uses standard **RGB Edges** ($M_{RGB}$).
    * [cite_start]The model uses these edges to guess where walls or tables continue behind occlusions[cite: 262].

---

### Step 4: The Core Repair (Diffusion Models)
Now you have two separate inputs and two separate condition maps.
* **Action:** Two separate **Diffusion Models** run in parallel.
    * They take the messy input (Glass or Background).
    * [cite_start]They compress it into a small "Latent Space" representation[cite: 208].
    * [cite_start]They add noise to destroy the data, then "denoise" it step-by-step to rebuild clean depth[cite: 120].
    * **Crucial:** During denoising, they look at the **Condition Maps ($M_{DA}$ or $M_{RGB}$)** to guide the reconstruction structure.

### Step 5: The Merge
* **Result A:** A clean depth map of just the glass/mirrors.
* **Result B:** A clean depth map of just the background.
* [cite_start]**Final Action:** Add Result A + Result B together[cite: 159].
* **Output:** One complete, clean depth image ($D_{out}$).



**Next Step:**
Now that you see the flow, would you like to start implementing the **Stage 1 (Region Proposal)** using TROSNet, or would you prefer to build the **Guidance Map generation** (Step 3) logic first?