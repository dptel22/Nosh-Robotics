# AI Assistance & Technical Troubleshooting Log

> **Project:** Nosh Robotics - Computer Vision Intern Assignment  
> **Repository:** `dptel22/Nosh-Robotics`  
> **Notebook Context:** `Task_1.ipynb` (SAM 3 Pretrained Model) & `Task_2.ipynb` (Classical Image Processing & Evaluation)  
> **Compliance Note:** In accordance with the assignment guidelines (*"AI tools: You can use AI coding tools. Briefly disclose where you've used them. You're expected to clearly understand and explain the underlying methods used."*), this document details all technical errors encountered during development, the underlying root causes, the AI prompts/assistance used, and the applied solutions.

---

## Summary of Incidents

| # | Error / Symptom | Pipeline Stage | Root Cause Category | Resolution |
|---|-----------------|----------------|---------------------|------------|
| 1 | `pkg_resources` BPE resolution failure (`sam3.__file__ is None`) | Task 1 Model Setup | Python 3.13 editable install behavior | Explicitly pass `bpe_path` to `build_sam3_image_model` |
| 2 | `403 Forbidden` / `LocalEntryNotFoundError` (`facebook/sam3`) | Task 1 Model Download | Hugging Face Gated Repo permissions | Accept license on HF & enable public gated access on token |
| 3 | `NameError: name 'image_paths' is not defined` | Task 1 Data Ingestion | Variable contract mismatch after refactor | Aliased `image_paths = all_images` |
| 4 | `NameError: name 'processor' is not defined` | Task 1 Model Inference | Colab out-of-order cell execution | Re-ran prerequisite "Load SAM 3" initialization cell |
| 5 | `RuntimeError: mat1 and mat2 must have the same dtype (BFloat16 vs Float)` | Task 1 Forward Pass | Model weights (`bfloat16`) vs inputs (`float32`) mismatch | Wrapped inference in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` |
| 6 | `TypeError: Got unsupported ScalarType BFloat16` | Task 1 Visualization | NumPy does not natively support PyTorch `bfloat16` | Explicitly cast tensors to float32 (`.float()`) before `.cpu().numpy()` |
| 7 | Pipeline Throughput Bottlenecks & T4 Hardware Crash Risk | Task 1 Code Review | Sequential I/O blocking GPU; hardcoded BF16 on Turing (T4); $O(N^2)$ NMS | DataLoader prefetching, ThreadPool async writes, hardware-safe AMP (`is_bf16_supported`), vectorized matrix IoU, crash-safe CSV deduplication |
| 8 | `FileNotFoundError: /content/out/masks` & Indentation Error | Task 2 Evaluation Setup | Outdated path reference (`MASKS`) and stale kernel function definition | Updated `get(i)` to use `mask_of` pointing to `/content/sam_masks`, fixed indentation, re-bound helpers in trial cell |
| 9 | Perceptual Hash Splitting & Data Leakage Risk | Task 2 Dataset Split | Potential cross-split leakage of identical scene captures | Automated pHash clustering (`THR=10`), verified 3 true duplicates + 1 false merge, kept threshold to prevent split reshuffling |
| 10 | Task 1 Pseudo-Ground Truth Anomaly (False Positives with 0 Missed Pixels) | Task 2 Error Analysis | Empty Task 1 masks on boiling water / extreme lighting scenes | Audited empty masks across all 160 images via Cell 6c, excluded empty masks from dev scoring, re-baselined T0/T1 |
| 11 | Naive Color Filtering Failures & "Largest Blob" Misclassification | Task 2 Algorithm Design | White-egg assumptions failed on brown eggs (`24`) & complex backgrounds (`108`, `28`, `39`) | Designed geometric `egg_score` (convexity $\times$ ellipse fit $\times$ centrality $\times$ area), integrated color-agnostic GrabCut mode & trimap refinement |

---

## Detailed Technical Breakdown

### Incident 1: BPE Vocabulary Path Resolution in Python 3.13 Editable Install

* **Error Message:** `pkg_resources` failed to locate the BPE vocabulary file path for SAM 3 (`sam3.__file__` resolves to `None`).
* **Stage:** SAM 3 model initialization (`build_sam3_image_model`).
* **Root Cause:**
  * In Python 3.13, PEP 660 editable installs (`pip install -e .`) use import hooks and loader mechanisms where `__file__` for namespace/editable packages can be `None`.
  * The legacy `pkg_resources` library relies on a physical `__file__` location to resolve bundled package resources (like tokenizer vocabularies). When `sam3.__file__` is `None`, path traversal fails.
* **Resolution:**
  * Bypassed `pkg_resources` automatic discovery by locating the physical vocabulary file path and explicitly supplying it via the `bpe_path` argument:
    ```python
    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH,
        bpe_path="/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    )
    ```
* **AI Disclosure:** AI identified why Python 3.13's editable installation broke `pkg_resources` and updated the initialization call to pass `bpe_path` directly.

---

### Incident 2: Hugging Face Gated Repository Permission (403 Forbidden)

* **Error Message:**
  ```text
  LocalEntryNotFoundError: 403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings to view this repository.
  Cannot access content at: https://huggingface.co/facebook/sam3/resolve/main/config.json
  ```
* **Stage:** Downloading official checkpoint/config from Hugging Face Hub.
* **Root Cause:**
  * The `facebook/sam3` repository is a **gated model repository** requiring explicit license agreement acceptance on Hugging Face.
  * Even when an authenticated Hugging Face API token (`HF_TOKEN`) is provided, **Fine-Grained Tokens** default to restricted repository access and do not automatically include permission for gated repositories unless explicitly checked.
* **Resolution:**
  1. Visited the Hugging Face model page (`huggingface.co/facebook/sam3`) and accepted the user terms.
  2. Updated the Hugging Face User Access Token settings to enable **"Read access to contents of all public gated repos"** (or used a classic Read token).
* **AI Disclosure:** AI diagnosed the 403 Forbidden error, identified that the repository is gated, and guided configuring fine-grained token permissions.

---

### Incident 3: Dataset Ingestion Variable Mismatch (`NameError: image_paths`)

* **Error Message:** `NameError: name 'image_paths' is not defined`.
* **Stage:** Loading unzipped dataset (`/content/egg.v3i.folder`).
* **Root Cause:**
  * The original dataset loading helper was refactored into a simpler script that saved scanned image paths into `all_images`.
  * Downstream segmentation and evaluation cells expected the variable name `image_paths`.
* **Resolution:**
  * Reconciled variable names by setting `image_paths = all_images`:
    ```python
    # Ensure downstream pipeline compatibility
    image_paths = all_images
    ```
* **AI Disclosure:** AI provided the simplified directory scanning logic and resolved the variable naming contract between notebook cells.

---

### Incident 4: Colab Out-of-Order Execution (`NameError: processor`)

* **Error Message:** `NameError: name 'processor' is not defined`.
* **Stage:** Running the test segmentation cell (`MXdMYBEgr4Px`).
* **Root Cause:**
  * In Google Colab / Jupyter interactive sessions, cells can be executed non-linearly.
  * The cell defining `processor` under "Load SAM 3" (cell `tIGJ93LXql-D`) had not been run yet in the active runtime kernel.
* **Resolution:**
  * Executed the "Load SAM 3" initialization cell first to instantiate the model and tokenizer/processor in memory before running inference.
* **AI Disclosure:** AI identified notebook cell execution order dependency and instructed running the initialization cell first.

---

### Incident 5: Matrix Multiplication Precision Mismatch (BFloat16 vs Float32)

* **Error Message:**
  ```text
  RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
  ```
* **Stage:** SAM 3 forward pass / inference (`segment_image`).
* **Root Cause:**
  * To conserve GPU VRAM and take advantage of modern Tensor Cores (e.g. NVIDIA Ampere / A100 / T4), SAM 3's backbone transformer weights are loaded in 16-bit Brain Floating Point (`torch.bfloat16`).
  * In contrast, the standard input preprocessors (image tensors and text prompt embeddings) produced 32-bit Single Precision (`torch.float32`) tensors.
  * PyTorch linear layers (`torch.nn.functional.linear` / `torch.matmul`) strictly require both matrices (`mat1` and `mat2`) to have matching data types.
* **Resolution:**
  * Wrapped the forward inference step inside PyTorch's Automatic Mixed Precision (`torch.autocast`) context manager, which handles type conversion automatically:
    ```python
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
    ```
* **AI Disclosure:** AI explained the mixed-precision tensor clash between model weights and input embeddings and wrapped the forward pass in `torch.autocast`.

---

### Incident 6: NumPy Incompatibility with PyTorch `bfloat16`

* **Error Message:**
  ```text
  TypeError: Got unsupported ScalarType BFloat16
  ```
* **Stage:** Post-processing masks and scores for visualization and OpenCV/NumPy masking.
* **Root Cause:**
  * When executing under `torch.autocast`, output tensors (segmentation masks and confidence scores) are returned in `torch.bfloat16`.
  * Standard NumPy (`numpy.ndarray`) does not natively support PyTorch's custom `bfloat16` scalar type.
  * Calling `.cpu().numpy()` on a `bfloat16` tensor causes PyTorch's bridge to raise a `TypeError`.
* **Resolution:**
  * Explicitly cast output tensors to 32-bit float (`.float()`) prior to bridging to NumPy:
    ```python
    masks_np = masks.float().cpu().numpy()
    scores_np = scores.float().cpu().numpy()
    ```
* **AI Disclosure:** AI diagnosed NumPy's lack of native `bfloat16` scalar support and updated the `segment_image` function to cast tensors to `.float()` before NumPy conversion.

---

### Incident 7: Production Optimization, T4 Hardware Safety, and Asynchronous Pipeline Refactoring

* **Context:** Senior CV code review identifying engineering bottlenecks, GPU idle time, Colab T4 hardware limitations, and resume logic edge cases.
* **Stage:** End-to-End Pipeline Optimization (`Task_1.ipynb` / `task_1.py`).
* **Root Causes & Solutions:**
  1. **Turing Architecture (T4) `bfloat16` Incompatibility:**
     * *Root Cause:* NVIDIA T4 GPUs (Compute Capability 7.5, Turing architecture) lack native hardware `bfloat16` Tensor Core support (introduced in Ampere SM 8.0). Hardcoding `dtype=torch.bfloat16` triggers slow CPU emulation or runtime crashes on free Colab instances.
     * *Resolution:* Dynamically detect hardware support:
       ```python
       amp_dtype = (
           torch.bfloat16
           if (device == "cuda" and torch.cuda.is_bf16_supported())
           else torch.float16 if device == "cuda" else torch.float32
       )
       ```
  2. **I/O Blocking & GPU Idling:**
     * *Root Cause:* Sequential loop interleaved Disk Read $\rightarrow$ CPU Decode $\rightarrow$ GPU Inference $\rightarrow$ CPU NMS $\rightarrow$ 3x Disk Writes, causing the GPU to sit idle during blocking image reads and PNG/JPEG compression.
     * *Resolution:* Decoupled I/O via a PyTorch `DataLoader` with `num_workers=2` (background image prefetching and RAM decoding) and a `ThreadPoolExecutor` with 2 background workers for non-blocking disk writes (`Image.save`).
  3. **Vectorized Mask NMS:**
     * *Root Cause:* Pairwise `np.logical_and(masks[i], masks[j]).sum()` loops in Python executed in $O(N^2)$ over full-resolution arrays.
     * *Resolution:* Vectorized pairwise intersections via matrix multiplication (`flat_masks @ flat_masks.T`), computing all $(N, N)$ IoUs in a single vectorized BLAS operation.
  4. **Resume Logic Deduplication:**
     * *Root Cause:* An append-only CSV write caused duplicate rows if an image was re-processed after a partial crash or if output files were deleted.
     * *Resolution:* Implemented in-memory record tracking (`completed_records`), verifying both CSV record presence and physical file existence, purging stale incomplete entries, and guaranteeing single-row uniqueness across resumes.
  5. **Instance Occlusion Ordering:**
     * *Root Cause:* Assigning non-overlapping instance IDs purely by confidence score allowed diffuse high-confidence masks to swallow smaller overlapping instances.
     * *Resolution:* Sorted masks by area descending, allowing smaller occluded instances to preserve their boundaries and pixels.
* **AI Disclosure:** AI synthesized the senior review feedback, verified the hardware and algorithmic bottlenecks against the repository, and implemented the refactored asynchronous pipeline and hardware-safe AMP checks.

---

### Incident 8: Task 2 Evaluation Setup Path Resolution (`FileNotFoundError`) & Stale Kernel State

* **Error Message:** `FileNotFoundError: [Errno 2] No such file or directory: '/content/out/masks/...'`
* **Stage:** Task 2 Evaluation Setup & Baseline Trial T0 Execution (Cells `K4b7UWm2TORi`, `_V8aqfi4TTSC`).
* **Root Cause:**
  * The dataset helper function `get(i)` was referencing an outdated global variable `MASKS` mapped to a non-existent `/content/out/masks` directory.
  * In the active Colab runtime, Task 1 ground-truth SAM masks were unzipped into `/content/sam_masks` and indexed in the `mask_of` dictionary.
  * In addition, an indentation error existed on the return statement block of `get(i)`, and the Jupyter notebook kernel retained a stale version of the function from an earlier cell (`Z1dZZRVITMbI`).
* **Resolution:**
  1. Updated `get(i)` inside evaluation cell `K4b7UWm2TORi` to dynamically fetch masks from `mask_of[imgs[i].stem]`, directing file reads to `/content/sam_masks`.
  2. Fixed the indentation error in `get(i)`.
  3. Explicitly redefined the helper functions within trial runner cell `_V8aqfi4TTSC` before execution, ensuring the active kernel used the corrected lookup logic.
  4. Executed baseline trial T0 (Otsu thresholding on grayscale), successfully logging **mIoU = 0.5334** (median IoU = 0.5648).
* **AI Disclosure:** AI identified the kernel-level state mismatch and path redirection necessity, corrected `get(i)` to use the `mask_of` registry, and fixed block indentation.

---

### Incident 9: Data Leakage Prevention via Perceptual Hash Group Split Audit

* **Topic:** Preventing data leakage between Dev (127 images) and Eval (33 images) splits in a 4:1 ratio.
* **Stage:** Task 2 Dataset Partitioning.
* **Context & Implementation:**
  * The assignment strictly mandates reducing data leakage between splits. To prevent burst-shot duplicates from leaking across splits, an automated perceptual hashing (`imagehash.phash`) clustering was implemented.
  * Near-duplicate images were grouped with a Hamming distance threshold of `THR=10`, ensuring all members of a cluster are placed into the same split.
  * A human-in-the-loop audit of the merged clusters revealed 4 pairs:
    * **3 True Duplicates:** `78 <-> 27` (distance 2), `51 <-> 45` (distance 6), and `60 <-> 8` (distance 6) — verified as multiple captures of the identical physical scene.
    * **1 False Merge:** `1 <-> 7` (distance 10) — distinct scenes (hand holding an egg vs. egg in greenery).
* **Technical Decision:**
  * Retained `THR=10`. In data partitioning, false merges are benign with respect to leakage: grouping two unrelated images together merely forces them onto the same side of the split without introducing cross-split contamination. In contrast, tightening the threshold to split `1` and `7` would alter the split composition, invalidating previously logged trial results.
* **AI Disclosure:** AI assisted in designing the pHash group-splitting logic, conducted visual cluster inspection, and provided the technical reasoning for retaining `THR=10`.

---

### Incident 10: Task 1 Pseudo-Ground Truth Anomaly Diagnosis (Empty Masks)

* **Symptom:** On dev split images `135`, `145`, `124`, and `82` (e.g. white egg submerged in boiling water), error overlays displayed false positives (red) with 0 missed pixels (blue), yet scored IoU = 0.
* **Stage:** Task 2 Evaluation & Error Analysis.
* **Root Cause:**
  * The evaluation script treats Task 1 SAM masks as ground truth. On these challenging images, SAM failed (`NO_DETECTION`), producing an empty (all-zero) mask.
  * When the ground-truth mask has zero positive pixels, any predicted foreground pixel yields $\text{Intersection} = 0$, resulting in an IoU of $0.0$ regardless of segmentation quality.
* **Resolution:**
  1. Authored QA audit cell `6c` to detect all empty masks across the 160-image dataset:
     ```python
     empty = [i for i in range(len(imgs)) if not get(i)[1].any()]
     ```
  2. Verified visually whether eggs were physically present.
  3. Filtered out verified empty-mask images from the dev evaluation set (`dev_i = [i for i in dev_i if get(i)[1].any()]`) and re-baselined trials (T0, T1) to ensure all subsequent trials are evaluated against valid, comparable ground truth.
  4. Documented this limitation of the Task 1 pseudo-ground truth key for the final report.
* **AI Disclosure:** AI diagnosed the lack of false-negative pixels as an indicator of empty ground truth and provided the QA audit and filtering scripts.

---

### Incident 11: Classical Segmentation Progression — Geometric Blob Scoring & GrabCut

* **Context:** Developing classical image processing segmentation to closely match Task 1 SAM quality across trials T0–T6.
* **Stage:** Task 2 Algorithm Design & Experimentation.
* **Findings & Failure Analysis:**
  * **Baseline T0 (Otsu threshold on grayscale):** Achieved mIoU = 0.5334 (median 0.5648).
  * **Trial T1 (HSV color range for "white egg"):** Resulted in a statistical tie with T0. Analysis revealed that the assumption of high value ($V$) and low saturation ($S$) failed on real-world dataset diversity:
    * Dark brown / mottled eggs (e.g., image `24`) were completely dropped by white-egg color thresholds.
    * Complex backgrounds (patterned dishes in `108`, concrete texture in `28`, printed text/packaging in `39`) exhibited high brightness, misleading naive color rules.
    * The default `select="largest"` heuristic misclassified large bright background regions as the egg.
* **Algorithmic Solutions:**
  1. **Multi-Factor Geometric Scoring (`select="score"` - T2):** Replaced "largest blob" with an engineered egg likelihood metric:
     $$\text{egg\_score} = \text{solid}(\text{comp}) \times \text{fit} \times \left(1 - 0.5 \times \text{dist}\right) \times \sqrt{\frac{\text{area}}{H \times W}}$$
     where:
     * $\text{solid}(\text{comp}) = \frac{\text{Contour Area}}{\text{Convex Hull Area}}$ (measures convexity; penalizes irregular or jagged backgrounds).
     * $\text{fit} = \frac{\min(\text{Area}, \text{Ellipse Area})}{\max(\text{Area}, \text{Ellipse Area})}$ (measures geometric fit to an fitted ellipse).
     * $\text{dist} = \frac{\sqrt{(c_x - w/2)^2 + (c_y - h/2)^2}}{\sqrt{(w/2)^2 + (h/2)^2}}$ (penalizes peripheral background regions).
     * $\sqrt{\text{Area} / (H \times W)}$ favors plausible object proportions over tiny noise specks.
  2. **Color-Agnostic GrabCut (`mode="grabcut"` - T3):** Initialized GrabCut with an outer border bounding rectangle (`gc_margin`), utilizing Gaussian Mixture Models (GMMs) to separate foreground from background without hardcoded color thresholds, successfully segmenting both white and brown eggs.
  3. **Morphological & Trimap Refinement (`refine="grabcut"`):** Built a three-region trimap from morphological erosion (sure foreground) and dilation (uncertain transition ring) to refine boundary contours.
* **AI Disclosure:** AI formulated the geometric `egg_score` equation, implemented the GrabCut bounding and trimap refinement modules, and structured the comparative trial progression.

---

## Interview Talking Points & Theoretical Takeaways

When discussing these issues during technical evaluation:

1. **Mixed Precision (`bfloat16` vs `float16` vs `float32`):**
   * `bfloat16` preserves the same 8-bit dynamic range (exponent) as `float32`, drastically reducing underflow/overflow risk compared to `float16` (which only has a 5-bit exponent), making it ideal for deep transformer models like SAM 3.
   * `torch.autocast` manages operations dynamically: compute-heavy operations (e.g. matrix multiplications, convolutions) execute in `bfloat16`, while precision-sensitive operations (e.g. softmax, reductions, loss calculations) remain in `float32`.

2. **NumPy & PyTorch Interoperability:**
   * NumPy's memory layout and C-API historically only supported standard IEEE 754 floats (`float16`, `float32`, `float64`).
   * Bridging non-standard deep learning types (`bfloat16`) to CPU NumPy always requires an explicit type cast (e.g., `.to(torch.float32).cpu().numpy()`).

3. **Packaging & Resource Management:**
   * Modern Python packaging (PEP 517/518/660) has moved away from `pkg_resources` (from `setuptools`) in favor of `importlib.resources`. Code relying on `pkg_resources` can break when packages are installed in editable mode (`-e`) or in Python >= 3.12/3.13.

4. **Data Leakage & Perceptual Hashing (pHash):**
   * Standard random train/test splits fail on robotics datasets containing consecutive video frames or burst photos because nearly identical scene distributions leak across splits.
   * Perceptual hashing (`pHash`) computes a frequency-domain Discrete Cosine Transform (DCT) fingerprint of each image. Grouping images within a Hamming distance threshold (`THR=10`) ensures that visual duplicate clusters are allocated atomically to either dev or eval, eliminating optimistic evaluation bias.

5. **GrabCut & Energy Minimization via Graph Cuts:**
   * Unlike static color thresholding (which fails when object colors vary from white to dark brown), GrabCut models foreground and background color distributions dynamically using two Gaussian Mixture Models (GMMs, typically 5 components each).
   * It frames segmentation as an energy minimization problem balancing a regional data term $U$ (color likelihood under the GMM) with a pairwise boundary smoothness term $V$ (penalizing color gradients across neighboring pixels), solved via min-cut/max-flow.

6. **Geometric Shape Priors vs. Greedy Blob Selection:**
   * Greedy heuristics like `largest_blob` fail in robotic kitchen environments where background structures (dishes, pans, countertops) are larger and brighter than the target egg.
   * Coupling convexity ($\text{Area} / \text{ConvexHull}$), second-order central moments ($\text{fitEllipse}$), and spatial camera-centering priors produces a robust multi-attribute decision score that reliably isolates the object of interest.

---

### Incident 12 / Task 4: Video Tracking Pipeline Implementation & Validation
* **Stage:** Task 4 (Video Egg Tracking).
* **AI Disclosure:** AI authored `egg_segmenter.py` (verbatim extraction of the frozen segmenter from `Task_3.ipynb` with `cv2.setRNGSeed(0)` and modular functions) and `task4_track.py` (constant-velocity 2D Kalman filter, Hungarian distance matching, table ROI cropping, morphological opening fallback for cutting text/arrow attachments, 32-bin grayscale histogram camera-cut detection, and diagnostic plotting).
* **Manual Verification Required by Human Developer:** Visually inspect `preview_before.png` vs `preview_after.png` and review the output video `tracked_eggs.mp4` to check by eye the suspect ID switch (ID 2 lost -> ID 10 born at ~30–33s) and verify that brief finger occlusions do not corrupt track continuity.
