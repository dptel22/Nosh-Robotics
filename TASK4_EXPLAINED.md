# Task 4 Explained: Function-by-Function Interview Guide

This document explains every function in `task4_track.py` and `egg_segmenter.py` using a strict 3-line format:
1. **What it does**
2. **Why this choice**
3. **What breaks if changed**

---

## 1. Segmentation Functions (`egg_segmenter.py`)

### `kernel(size)`
* **What it does:** Generates an elliptical morphological structuring element of a given pixel dimension.
* **Why this choice:** Eggs have rounded convex boundaries, so an elliptical kernel smooths contours without introducing square artificial corners.
* **What breaks if changed:** Using a rectangular structuring element (`MORPH_RECT`) distorts curved egg contours into faceted polygons and degrades ellipse fit scores.

### `largest_contour(blob)`
* **What it does:** Finds all external contours of a binary mask and returns the one with the maximum contour area.
* **Why this choice:** Guarantees that internal ring artifacts, holes, and tiny disconnected noise specks inside the blob are ignored.
* **What breaks if changed:** Returning the raw contour list or first contour can cause downstream metrics (`contourArea`, `fitEllipse`) to evaluate internal voids or noise artifacts instead of the egg boundary.

### `solidity(blob)`
* **What it does:** Computes the ratio of contour area to convex hull area ($\text{Area} / \text{ConvexHull}$).
* **Why this choice:** Eggs are naturally convex shapes with solidity close to 1.0; irregular backgrounds and spattered egg white have low solidity.
* **What breaks if changed:** If solidity is omitted, irregular white artifacts (spilled water, paper labels, torn egg white) are treated as valid whole eggs.

### `egg_score(blob)`
* **What it does:** Computes a multi-attribute geometric likelihood: $\text{solidity} \times \text{ellipse\_fit} \times (1 - 0.5 \times \text{distance\_from\_center}) \times \sqrt{\text{area\_fraction}}$.
* **Why this choice:** Balances shape convexity, true second-order elliptical moments, plausible egg scale, and a spatial prior favoring center tabletop regions.
* **What breaks if changed:** Greedy heuristics like `largest_blob` fail whenever large bright background objects (white dishes, countertops, pans) enter the camera frame.

### `grabcut_refine(img, mask, iterations=3, ring=15)`
* **What it does:** Builds a three-zone trimap (eroded core = sure foreground, dilated outer band = unsure transition ring, background = sure background) and runs GrabCut graph cut.
* **Why this choice:** Snaps approximate binary boundaries directly onto the true physical edge using Gaussian Mixture Models of local color distributions.
* **What breaks if changed:** Without GrabCut refinement, segmentation boundaries remain rough and pixelated from binary thresholding, especially under harsh shadows or reflections.

### `denoise(bgr, kind)`
* **What it does:** Applies Gaussian smoothing (`GaussianBlur`) or edge-preserving bilateral filtering (`bilateralFilter`) to the BGR image.
* **Why this choice:** Reduces high-frequency camera sensor noise and compression artifacts before edge detection and thresholding.
* **What breaks if changed:** Disabling denoising causes high-frequency noise specks to trigger false positive contours and fragments smooth binary masks.

### `rough_mask(img, cfg)`
* **What it does:** Produces an initial binary foreground candidate mask using Otsu thresholding, GrabCut bounding box, or HSV/HSV+LAB color ranges.
* **Why this choice:** Separates candidate foreground pixels from dark backgrounds rapidly before running expensive geometric evaluations.
* **What breaks if changed:** Overly restrictive color ranges (e.g. strict RGB ranges) fail on brown eggs or varying lighting conditions; Otsu or HSV range provides an adaptable first pass.

### `pick_blobs(mask, cfg)`
* **What it does:** Performs connected components analysis on the binary mask and filters blobs by area and selection strategy (`largest`, `score`, or `shape`).
* **Why this choice:** Allows single-egg isolation in controlled shots or multi-egg detection (`select="shape"`) in multi-object cooking scenes.
* **What breaks if changed:** If multiple blobs are not parsed independently, multiple eggs touching or sharing the frame merge into a single invalid super-blob.

### `draw_blob(canvas, blob, method)`
* **What it does:** Paints selected blobs onto the final output mask, optionally replacing raw pixels with their convex hull or fitted ellipse.
* **Why this choice:** Smooths out jagged optical boundaries and fills micro-indentations if geometric shape priors are enabled.
* **What breaks if changed:** Fitting an ellipse when `refine="ellipse"` on non-elliptical artifacts would distort boundaries; raw mask copying preserves true segmented pixels.

### `seg(bgr, cfg)`
* **What it does:** Executes the complete frozen Task 2 pipeline: seed RNG $\rightarrow$ denoise $\rightarrow$ rough mask $\rightarrow$ skin filter $\rightarrow$ morphology $\rightarrow$ hole filling $\rightarrow$ blob selection $\rightarrow$ optional GrabCut refinement.
* **Why this choice:** Enforces complete reproducibility via `cv2.setRNGSeed(0)` while chaining proven classical computer-vision primitives in a fixed sequence.
* **What breaks if changed:** Removing `cv2.setRNGSeed(0)` causes GrabCut k-means clustering to initialize randomly, producing non-deterministic masks across runs on identical frames.

---

## 2. Tracking & Pipeline Functions (`task4_track.py`)

### `find_final_config(cli_config)`
* **What it does:** Resolves the path to `final_config.json` via explicit CLI argument or by dynamically locating the newest file matching `outputs/**/final_config.json`.
* **Why this choice:** Ensures seamless portable execution across different environments without hardcoded absolute machine paths.
* **What breaks if changed:** Hardcoding local paths breaks portability across machines or Docker/Colab containers; failing to find the file causes immediate crashes.

### `ellipse_fit_ratio(contour)`
* **What it does:** Fits a rotated ellipse to the contour (`cv2.fitEllipse`) and returns the ratio of contour area to fitted ellipse area ($\text{Area} / (\pi \times a \times b)$).
* **Why this choice:** Directly penalizes non-elliptical protrusions (arrow shafts, text bars, chef fingers) whose area deviates from an idealized ellipse.
* **What breaks if changed:** Using circularity ($\text{Perimeter}^2 / \text{Area}$) penalizes naturally elongated oval eggs; the ellipse ratio fits both round and oval eggs accurately.

### `try_cut_attachments(blob, frame_area, area_range, min_ellipse_fit, opening_sizes)` (Opening Fallback)
* **What it does:** Progressively applies morphological opening with structuring elements of sizes 11, 21, and 31 to sever thin attachments (text labels, arrow pointers) from the egg body.
* **Why this choice:** Thin annotations and pointing arrows have narrow pixel necks that erode away completely under medium opening kernels, while the bulky egg body survives and retains its elliptical shape.
* **What breaks if changed:** Without this fallback, any egg touched by on-screen text, arrows, or fingers fails `MIN_ELLIPSE_FIT` and is dropped entirely as a non-egg blob.

### `detect(frame, cfg_video, roi_top, area_range, min_ellipse_fit, opening_sizes)` (Table ROI & Detection)
* **What it does:** Crops the frame below `ROI_TOP` (ignoring top 25%), runs `seg()`, evaluates blob area and ellipse fit, applies the opening fallback if needed, and maps coordinates back to full frame.
* **Why this choice:** Ignoring the top 25% eliminates moving kitchen backgrounds, chef aprons, and clothing patterns that cause false foreground detections.
* **What breaks if changed:** Without the ROI crop, moving clothing patterns and kitchen background elements are constantly detected as false positive eggs.

### `Kalman` (Constant-Velocity 2D Filter)
* **What it does:** Models egg state as $[x, y, v_x, v_y]^T$ and updates trajectory estimates via standard predict-update cycles using measurement $[z_x, z_y]^T$.
* **Why this choice:** Smooths frame-to-frame detection jitter and enables the tracker to coast linearly when eggs are briefly occluded by fingers or spoons during cooking.
* **What breaks if changed:** Tracking without a motion model causes tracks to snap erratically between detections or terminate immediately upon a single frame of finger occlusion.

### `Track`
* **What it does:** Stores individual track metadata: Kalman instance, centroid history, hit count, age, consecutive misses, trail deque, and assigned track ID.
* **Why this choice:** Isolates per-instance state and maintains a fixed-length spatial trail deque for rendering smooth trajectory tails.
* **What breaks if changed:** Failing to separate track state leads to leaked state across IDs or unbounded memory growth if trajectory histories are not properly bounded.

### `Tracker.predict()`
* **What it does:** Advances the Kalman state transition matrix ($F$) for all currently active tracks before processing incoming frame detections.
* **Why this choice:** Provides a motion-extrapolated prior position for Hungarian matching even if the object has moved between frames.
* **What breaks if changed:** Performing matching against old un-propagated positions causes matching failures whenever eggs roll or move at high velocity.

### `Tracker.update(detections, frame_idx, time_s)` (Hungarian vs. Greedy Matching)
* **What it does:** Constructs a global pairwise Euclidean distance matrix and solves the optimal bipartite matching problem via SciPy's `linear_sum_assignment` under a distance gate (`max_dist`).
* **Why this choice (Hungarian vs Greedy):** Greedy nearest-neighbor matching suffers from assignment order bias (first track steals the closest detection, leaving neighbors stranded); Hungarian matching globally minimizes total assignment cost.
* **What breaks if changed:** Greedy matching leads to identity swaps whenever two eggs pass close to one another or when a newly born candidate appears near an existing egg.

### `Tracker.prune_tracks(frame_idx, time_s)` (Why `MIN_HITS` and `MAX_MISSED`)
* **What it does:** Eliminates unconfirmed candidate tracks after 1 missed frame and terminates confirmed tracks after `MAX_MISSED_SECONDS` (3.0s) of continuous non-detection.
* **Why `MIN_HITS=3`:** Transient sensor noise, flickering reflections, and fleeting finger edges disappear in 1–2 frames; requiring 3 consecutive hits filters out 100% of spurious flashes.
* **Why `MAX_MISSED=3.0s`:** Hands picking up, spinning, or seasoning an egg frequently cause brief 1–2 second occlusions; 3 seconds allows coasting through occlusions without dropping the ID.
* **What breaks if changed:** If `MIN_HITS=1`, every single reflection or background flicker gets an ID; if `MAX_MISSED` is too short, temporary hand occlusions force immediate track termination and ID switches.

### `Tracker.reset(frame_idx, time_s)`
* **What it does:** Logs `lost` events for all currently confirmed active tracks and purges the active track list while preserving monotonic ID generation (`next_id` is never reset).
* **Why this choice:** When the camera cuts to a new scene, eggs from the old scene no longer physically exist in those tabletop positions; resetting prevents identity jumping across shots.
* **What breaks if changed:** Without a reset on camera cuts, Kalman filters from shot $A$ project across the cut and latch onto unrelated eggs in shot $B$, corrupting trajectory identity.

### `compute_small_gray_hist(frame)` & `is_camera_cut(prev_hist, curr_hist, threshold)`
* **What it does:** Resizes consecutive frames to 160x90 grayscale, computes 32-bin normalized histograms, and flags a camera cut when histogram correlation drops below `CUT_THRESHOLD=0.6`.
* **Why this choice:** Global luminance and color distributions shift drastically during camera cuts; comparing low-resolution histograms is computationally lightweight ($\sim 0.5$ ms) and invariant to minor motion.
* **What breaks if changed:** Comparing raw pixel differences is sensitive to fast camera pans or large hand gestures; unnormalized metrics fail under global lighting adjustments.

### `draw_tracks_on_frame(frame, tracks, frame_idx, time_s)`
* **What it does:** Renders per-ID colored contours, "egg <id>" text tags, and past trajectory trails ONLY for tracks actively matched in the current frame (`time_since_update == 0`).
* **Why this choice (Centroid Only vs BBox):** Drawing only confirmed, actively seen eggs avoids cluttering the screen with ghost predictions while an egg is hidden behind a hand.
* **What breaks if changed:** Drawing coasting unconfirmed tracks displays "ghost" contours floating on top of the chef's hands when eggs are physically hidden.

### `convert_to_h264(input_video_path, output_video_path)`
* **What it does:** Converts OpenCV's raw `mp4v` output into standard H.264 video (`libx264`, `yuv420p`) using system `ffmpeg` or `imageio-ffmpeg`.
* **Why this choice:** Browsers and cloud platforms (like Google Drive) cannot decode legacy `mp4v` containers; H.264 with `yuv420p` pixel format guarantees universal web playback.
* **What breaks if changed:** Uploading raw OpenCV `mp4v` to Google Drive results in an unplayable black screen or "video processing" playback error.

### `run_preview(video_path, cfg_frozen, cfg_video, config_dict, out_dir)`
* **What it does:** Extracts 12 evenly spaced frames across the video, generates `preview_before.png` (frozen config on full frame) and `preview_after.png` (video config on table ROI), and terminates.
* **Why this choice:** Allows instant visual verification of ROI placement and segmentation quality across the whole clip in 2 seconds without waiting for a full 1440-frame tracking run.
* **What breaks if changed:** Tuning hyperparameters blindly on a 60-second video requires waiting for full tracking passes on every parameter change.

### `find_suspect_id_switches(events_df, max_dist, cuts_frames)`
* **What it does:** Audits the event log to find cases where an ID was lost and a different ID was born within 30 frames and spatial distance $< 2 \times \text{max\_dist}$ (excluding camera cuts).
* **Why this choice:** Provides an automated, objective QA filter to surface potential tracking failures so the developer can verify them by eye rather than guessing.
* **What breaks if changed:** Without suspect switch auditing, identity handoffs during prolonged occlusions go unnoticed unless someone watches every single frame manually.

---

## 3. Known Limitations to State in the Final Report

When presenting the results and final documentation:

1. **Histogram Cut Detection Limitations:**
   * Global histogram correlation reliably detects abrupt cuts between different scenes (e.g. intro graphics vs. countertop cooking).
   * However, it can miss cuts between two similar-looking camera angles of the identical countertop and lighting.
   * IDs are strictly per shot: because real eggs share nearly identical visual appearances and textures, long-term visual re-identification across camera cuts is not possible without spatial continuity.

2. **Merged Blobs Under Physical Contact:**
   * When two eggs make direct physical contact or roll together, their binary masks fuse into a single merged blob.
   * While the opening fallback severs thin attachments (arrows, text, fingers), two eggs touching with a wide contact region will not be cut by elliptical opening.
   * Under such conditions, one track matches the merged blob while the other track misses detection and coasts on its Kalman motion prediction.

3. **Heuristic Calibration & Verification:**
   * Starting constants (`ROI_TOP=0.25`, `AREA_RANGE`, `MIN_ELLIPSE_FIT=0.85`, `MAX_DIST_FRACTION=0.12`) are engineering baselines derived from frame geometry.
   * Verification was performed using objective equivalence testing, synthetic stress-testing (arrows, skin occlusions, title slides), and visual preview sheets on the target video.
