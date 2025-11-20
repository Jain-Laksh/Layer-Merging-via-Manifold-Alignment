# Beyond Naive Merging: Enhancing LLM Compression via Alpha Optimization, Task-Specific Similarity, and Neural Alignment ([Website](https://nishchaybhutoria.github.io/cs613-project-website))

> This is a course project for **CS 613: Natural Language Processing** at **IIT Gandhinagar**.
>
> **Base Paper:** Liu, D., et al. (2024). *Pruning via merging: Compressing LLMs via manifold alignment based layer merging*. [arXiv:2406.16330](https://arxiv.org/abs/2406.16330).
>
> **This Repository:** `https://github.com/Jain-Laksh/Layer-Merging-via-Manifold-Alignment`

---

## Overview

While Large Language Models (LLMs) have demonstrated remarkable capabilities, their massive size presents significant deployment challenges. This project builds on the "Pruning via Merging" (MKA) paper, which proposes a novel compression technique by merging similar layers based on manifold alignment.

The original MKA method, however, relies on several potentially suboptimal heuristics. This project investigates and provides solutions for three key limitations of the MKA baseline, using the **Llama3-8B** model as our testbed.

Our three main contributions are:

1.  **Optimizing Merge Weight ($\alpha$):** The MKA baseline sets the crucial layer merge weight ($\alpha$) using a simple similarity score. We treat $\alpha$ as a trainable parameter and optimize it using gradient descent and Bayesian optimization to find a more effective value.
2.  **Task-Specific Similarity:** The baseline assumes a static, task-independent layer similarity. We demonstrate that layer similarity is *not* static but is highly dependent on the **task domain** (e.g., MMLU humanities vs. math) and **language** (e.g., English vs. Spanish vs. Chinese).
3.  **Neural Alignment:** The baseline "naively averages" layer weights, which can merge functionally distinct neurons and degrade performance. We implement a robust **"align-then-merge" pipeline** that uses optimal permutation (via the Hungarian algorithm) to align functionally equivalent neurons *before* averaging them.

For a detailed background on the project's motivation and the MKA baseline, please see [this report](https://github.com/Jain-Laksh/NLP-Assignment1/blob/main/NLP_Assignment1_Report.pdf) . The full methodology and results of our three experiments are in `NLP_Assignment_2_Report.pdf`.

---

## Key Results

Our experiments produced three primary findings:

### 1. Alpha Optimization

We found that the MKA paper's simple heuristic ($\alpha=S_{lm}$) is surprisingly robust. While our optimized methods achieved a negligible improvement in final MMLU accuracy (e.g., **0.64888** for Bayesian Optimization vs. **0.64710** for the baseline), we discovered that the *optimal* $\alpha$ values are far more complex than the simple heuristic suggests.

The learned $\alpha$ values were highly variable (e.g., GD values oscillated between ~0.50 and ~0.71) and showed only a moderate positive correlation (0.560) with the similarity score. This proves that while the MKA heuristic is directionally correct, the optimal merge weight is influenced by other complex factors.

### 2. Task- and Language-Dependent Similarity

We quantitatively confirmed our hypothesis that layer similarity is not static.

* **Task-Dependence:** We generated similarity heatmaps for different MMLU domains and found clear visual and quantitative differences. For instance, the similarity patterns for "Math" and "Computer Science" were highly correlated (**0.951**), while "Legal" and "Humanities" also showed similar patterns.
* **Language-Dependence:** The effect was even more pronounced for language. When analyzing the same "medical" task in different languages, the similarity patterns diverged significantly. The correlation between Spanish (es) and Chinese (zh) similarity matrices was only **0.730**, confirming that layer redundancy is sensitive to the input language.

### 3. Neural Alignment

Our "align-then-merge" pipeline (Experiment 3) proved to be a more robust and reliable method for layer fusion.

The MKA baseline's performance was highly volatile, suffering a catastrophic drop in accuracy (from 0.662 to 0.547) at 12.5% compression. In contrast, our alignment method showed a smooth, predictable degradation and **avoided this intermediate drop**.

At that 12.5% compression level, our neural alignment method **outperformed the MKA baseline by +0.0864 MMLU accuracy** (0.6334 vs 0.5470). At higher compression ratios (40.625%), both methods converged to a similar accuracy, but our method's stability makes it a more reliable approach.

---

## Repository Structure

This repository contains all code and reports for the project.

* `/Updating Alpha/`: Code for **Experiment 1**. Contains scripts to optimize the $\alpha$ parameter using Gradient Descent (`optimize_alphas.py`) and Bayesian Optimization (`alpha_pipeline.py`).
* `/data_dependent_merging/`: Code for **Experiment 2**. Includes notebooks (`mmlu_and_multilingual.ipynb`) and scripts (`compute_similarity.py`) to generate and analyze task- and language-specific similarity heatmaps.
* `/Neural Alignment/`: Code for **Experiment 3**. Implements the full "align-then-merge" pipeline (`pipeline.py`) and evaluation notebook (`evaluate.ipynb`).
* `/data/` (within experiment folders): Contains MMLU dev and test data samples used for calibration and evaluation.
* `NLP_Assignment_2_Report.pdf`: The final project report detailing the methodology and results of our three experiments.

---

## Authors

* Aditya Borate 
* Aryan Solanki 
* Laksh Jain 
* Nishchay Bhutoria 
* Parthiv Patel 
* Rudra Pratap Singh 
* Soham Gaonkar 

---

## Acknowledgments

We would like to thank our professor [Dr. Mayank Singh](https://mayank4490.github.io/) and our project mentor [Sailesh Panda](https://saileshp97.github.io/) for their invaluable guidance, weekly feedback, and insightful suggestions, which significantly shaped the direction of this work. We also acknowledge the authors of the MKA paper for providing a strong and interesting foundation for our research.
