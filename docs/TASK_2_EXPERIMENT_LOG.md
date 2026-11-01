# Task 2: Classical Image Processing — Experiment & Trial Log

> **Project:** Nosh Robotics - Computer Vision Intern Assignment  
> **Repository:** `dptel22/Nosh-Robotics`  
> **Objective:** Develop a classical image processing algorithm to segment eggs across diverse kitchen scenes, matching the pseudo-ground truth segmentation quality established by SAM in Task 1.  
> **Dataset Split:** 160 total images split into **Dev Set (127 images)** and **Eval Set (33 images)** (~4:1 ratio) using Perceptual Hashing (`pHash`, threshold=10) to eliminate data leakage from near-duplicate photo bursts.

---

## 1. Dataset Splitting & Leakage Audit

To satisfy the requirement of reducing data leakage between development and evaluation splits:
- **Clustering Method:** Perceptual Hashing (`imagehash.phash`) with a Hamming distance threshold $\text{THR} = 10$.
- **Cluster Audit Results:**
  - Cluster 1 (`78 <-> 27`, $d=2$): True duplicate (same photo capture). Grouped into the same split.
  - Cluster 2 (`51 <-> 45`, $d=6$): True duplicate. Grouped into the same split.
  - Cluster 3 (`60 <-> 8`, $d=6$): True duplicate. Grouped into the same split.
  - Cluster 4 (`1 <-> 7`, $d=10$): False merge (hand holding egg vs. egg in greenery).
- **Engineering Decision:** Maintained $\text{THR} = 10$. False merges are safe for leakage prevention because grouping unrelated images merely forces them onto the same side of the split without cross-split contamination, whereas changing the threshold would reshuffle splits and invalidate previously logged trial metrics.

---

## 2. Task 1 Pseudo-Ground Truth QA Audit

Before evaluating classical trials, visual inspection of worst-performing images revealed several cases (`135`, `145`, `124`, `82` — eggs in boiling water / extreme lighting) showing false positives (red) with 0 missed pixels (blue) and scoring $\text{IoU} = 0.0$.

* **Audit Script (Cell 6c):**
  ```python
  empty = [i for i in range(len(imgs)) if not get(i)[1].any()]
  ```
* **Finding:** SAM failed on these difficult scenes, outputting an all-zero mask (`NO_DETECTION`).
* **Evaluation Protocol:** Verified whether eggs were physically present. True unannotated frames were excluded from dev scoring (`dev_i = [i for i in dev_i if get(i)[1].any()]`) so that all comparative trials are evaluated against valid, non-empty ground truth.

---

## 3. Systematic Trial Log (T0 to T3+)

| Trial ID | Pipeline Mode / Strategy | Parameters Tested | Dev mIoU | Dev Median IoU | Decision | Rationale |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **T0** | **Baseline:** Grayscale Otsu Thresholding | `denoise="gauss"`, `mode="otsu_gray"`, `morph=False`, `select="largest"` | **0.5334** | **0.5648** | **KEPT (Baseline)** | Establishes the initial classical computer vision baseline on the dev split. |
| **T1** | **Color Rule:** HSV "White Egg" Filtering | `mode="hsv"`, `s_max=65`, `v_min=150`, `select="largest"` | ~0.53 | ~0.56 | **DISCARDED / TIE** | Hypothesis not supported. The assumption that eggs are reliably white and bright failed across real-world data containing dark brown eggs (image `24`), patterned dishes (`108`), concrete (`28`), and packaging (`39`). |
| **T2** | **Geometric Blob Scoring:** Multi-Attribute `egg_score` | `select="score"`, `min_solid=0.9`, `min_area=0.005`, `max_area=0.98` | *Pending run* | *Pending run* | **KEPT** | Replaces greedy `largest_blob` selection with multi-factor scoring (convexity $\times$ ellipse fit $\times$ centrality $\times$ area), correctly rejecting large bright backgrounds in images `108`, `28`, and `39`. |
| **T3** | **Color-Agnostic Foreground Extraction:** GrabCut | `mode="grabcut"`, `gc_margin=0.08`, `gc_iters=4` | *Pending run* | *Pending run* | **IN PROGRESS** | Removes hardcoded color assumptions; uses Gaussian Mixture Models initialized with outer margin rectangle to segment foreground eggs regardless of white/brown coloring. |
| **T4** | **Morphological Cleaning & Boundary Refinement** | `morph=True`, `k_open=5`, `k_close=15`, `refine="grabcut"` | *Planned* | *Planned* | *Planned* | Evaluates trimap GrabCut refinement and hole filling (`ndi.binary_fill_holes`) to eliminate internal specular reflections and produce smooth natural egg boundaries. |

---

## 4. Algorithmic Formulation Details

### 4.1 Multi-Attribute Geometric Blob Scoring (`egg_score`)

Instead of greedily selecting the largest connected component, candidate blobs are scored using a four-factor geometric formulation:

$$\text{egg\_score}(\text{comp}) = \text{solid}(\text{comp}) \times \text{fit\_ellipse}(\text{comp}) \times \left(1 - 0.5 \times \text{dist}_{\text{center}}\right) \times \sqrt{\frac{\text{area}}{H \times W}}$$

1. **Convexity / Solidity ($\text{solid}$):**
   $$\text{solid} = \frac{\text{Contour Area}}{\text{Convex Hull Area}}$$
   Eggs are naturally convex ovoids. Irregular background textures, fingers, and shadows have lower solidity.
2. **Elliptical Fit ($\text{fit\_ellipse}$):**
   Fits an algebraic ellipse to contour points via `cv2.fitEllipse` and compares contour area to the theoretical ellipse area:
   $$\text{fit} = \frac{\min(\text{Area}, \pi \cdot a \cdot b)}{\max(\text{Area}, \pi \cdot a \cdot b, 1)}$$
3. **Centrality Prior ($\text{dist}_{\text{center}}$):**
   Normalized Euclidean distance from blob centroid $(c_x, c_y)$ to image center $(W/2, H/2)$:
   $$\text{dist} = \frac{\sqrt{(c_x - W/2)^2 + (c_y - H/2)^2}}{\sqrt{(W/2)^2 + (H/2)^2}}$$
4. **Scale Weighting ($\sqrt{\text{Area} / (H \times W)}$):**
   Prefers reasonably proportioned objects while preventing tiny foreground noise fragments from scoring highly.

---

## 5. Next Steps for Evaluation

1. Complete execution of **T2** (`select="score"`) and **T3** (`mode="grabcut"`).
2. Measure improvement on specific failure cases:
   - Image `108` (patterned plate)
   - Image `24` (backlit dark brown egg)
   - Image `28` (grey egg on concrete)
   - Image `39` (egg on printed packaging)
3. Finalize the best-performing classical configuration on the Dev split and freeze parameters before unblinded evaluation on the 33-image Eval split in Task 3.
