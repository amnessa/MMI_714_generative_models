Okay, let's break down this homework. It's about comparing different ways to measure the "distance" or difference between probability distributions using simple 1D data. 📊

Using a VSCode notebook is a great choice for this! You can combine your Python code, plots, and explanations all in one place. You'll primarily need libraries like `numpy` for numerical operations, `matplotlib.pyplot` for plotting, and `scipy.stats` for generating distributions and possibly calculating distances.

---

## Histograms, Bins, and Bin Edges

Think about making a **histogram**. You take a bunch of data points (like exam scores) and group them into ranges to see how many fall into each range.

* **Bin**: Each range or interval is called a **bin**. For example, scores 90-100 could be one bin (Grade A), 80-89 another (Grade B), and so on.
* **Bin Edge**: The numbers that define the start and end of each bin are the **bin edges**. For the Grade B bin (80-89), the bin edges are 80 and 90 (assuming 90 is the start of the next bin). The homework asks you to use the *same* bin edges for both distributions when comparing them to ensure a fair comparison.


---

## Distributions Explained

The homework asks you to compare a **Gaussian distribution** to a **non-Gaussian** one.

* **Gaussian (Normal) Distribution**: This is the classic "bell curve".  It's symmetric around its center (the mean) and its spread is defined by the standard deviation. Lots of natural phenomena approximate this shape. You can create one easily using `numpy.random.normal(mean, std_dev, size)`.
* **Non-Gaussian Distributions**: These are basically *any* other shape! The homework suggests a few examples:
    * **Two-peak mixture (Bimodal)**: Imagine mixing data from two different Gaussians (e.g., heights of adult men and heights of adult women combined). The resulting histogram would likely have two peaks.  You could create this by combining samples from two different `numpy.random.normal` calls.
    * **Uniform Distribution**: Every value within a certain range is equally likely. The histogram looks flat.  Use `numpy.random.uniform(low, high, size)`.
    * **Student's t-Distribution**: Looks similar to a Gaussian but has "heavier tails," meaning extreme values are a bit more likely.  Use `numpy.random.standard_t(df, size)` (where `df` is degrees of freedom).

---

## What the Homework Asks (Overall Goal)

The main goal is to **explore how different mathematical tools measure the difference between shapes of data distributions** and how sensitive these tools are to things like the number of bins you use or the amount of data you have. You'll be calculating and comparing:

1.  **Kullback-Leibler (KL) Divergence**: $KL(P||Q)$ and $KL(Q||P)$. Measures how one distribution diverges from a second, expected distribution. It's **not symmetric** ($KL(P||Q) \neq KL(Q||P)$).
2.  **Jensen-Shannon (JS) Divergence**: $JS(P,Q)$. A **symmetric** version of KL divergence, bounded between 0 and 1.
3.  **Wasserstein-1 (W1) Distance**: $W_{1}(P,Q)$. Also known as the Earth Mover's Distance. Intuitively, it's the minimum "cost" or "work" required to transform one distribution shape into the other, like moving piles of dirt. It's particularly good at comparing distributions that don't overlap.

You'll investigate:

* **Visual Differences**: How do the histogram shapes look different? (Problem 1)
* **Effect of Binning**: How do the calculated distance values change if you use fewer or more bins in your histograms? (Problem 2) You'll also need **smoothing** (adding a tiny value to all bin counts) to prevent errors like $log(0)$ when calculating KL and JS.
* **Metric Disagreements**: Can you create situations where W1 suggests distributions are very different while JS suggests they are similar, or vice-versa? (Problem 3)
* **CDF and W1**: How does the visual difference between the Cumulative Distribution Functions (CDFs) relate to the W1 distance? (Problem 4) The W1 distance is actually the area between the two CDF curves.
* **Effect of Sample Size**: How do the distance values change as you use more or fewer data points ($N$) to generate your distributions? Which metrics stabilize faster? (Problem 5)

Let me know when you want to dive into the specifics of calculating these metrics or tackling a particular problem! Good luck! 👍