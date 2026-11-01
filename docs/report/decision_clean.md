# Decision log — egg segmentation and tracking (Tasks 1–4)

Dataset: Roboflow export "egg.v3i.folder" (fire-bf7qe/egg-vzhtr), 160 images at 640×640, Roboflow folders train/valid/test = 112/32/16 (70/20/10).

## Task 1 — pseudo ground truth with SAM 3

- Planned a GroundingDINO+SAM two-model stack first; chose **SAM 3** instead because it cannot run locally and replaces both models (text prompt → mask directly). Ran on Google Colab (T4).
- Prompt **"egg"**, confidence threshold **0.30**, GPU mask-NMS (area window 0.002–0.98, IoU 0.70) with masks painted in reverse confidence order so overlapping instances keep separate labels.
- First pass left 10 images with no detection. **Patched**: retried those with fallback prompts ("egg", "boiled egg", "brown egg", "white egg", "oval object") at threshold 0.12, keeping the best mask. 7 of 10 recovered (flagged LOW_SCORE_RECOVERY); 3 stayed empty (not_damaged_42, _77, _116).
- Outputs: instance-label masks, union masks, overlays, sam_report.csv (160 rows) for all images in all three Roboflow folders.

## Task 2 — split and classical segmenter

- Roboflow's folders are not 4:1 and mix burst shots across folders, so I built my **own split**: perceptual-hash groups (Hamming ≤ 10, or identical base name) with GroupShuffleSplit, test 20%, seed 42 → **dev 127 / eval 33**, no group crosses the split; closest dev–eval pHash distance is 16.
- Trials on dev (log in experiments.csv): **T0** grayscale+Otsu+largest-blob baseline = mIoU 0.5558; **T1** HSV white-egg rule = 0.5588 (tie, lower median/recall, blind to dark eggs) — discarded; **T2** geometric egg-score selection = 0.5462 — discarded; **T3** GrabCut = 0.5402 at ~114× the runtime — discarded.
- **Froze the T0 baseline** as final_config.json (every tested alternative was worse or tied on dev) and generated pred_masks for all 160 images.

## Task 3 — evaluation

- Task 1 masks = answer key, frozen segmenter = prediction, same split.json. Eval scored **once**, after freezing; dev used for all tuning.
- Empty-key images (3) stay in the CSVs but are excluded from averages (dev n=125, eval n=32).
- Metrics: IoU/Dice/precision/recall + boundary F1(3 px)/HD95 + instance P/R/F1/count error (Hungarian matching, tp at IoU ≥ 0.5; key instances from SAM label IDs so touching eggs stay separate).
- Eval: mean IoU 0.615 (95% CI 0.531–0.692), Dice 0.729, boundary F1 0.240, HD95 188 px; dev mean IoU 0.565; instance eval P 0.812 / R 0.605. Boundaries and multi-egg scenes are the weak points.

## Task 4 — video tracking

- Reused the frozen segmenter (extracted verbatim into egg_segmenter.py, RNG seeded). On the video the frozen Otsu config masks the bright shirt/hands instead of eggs, so the video config overrides it: **HSV rule (V≥170, S≤70), morphological close, select="shape"**, min_solid 0.0 so merged blobs survive until detect() can cut them apart. Judged by eye on preview frames of the same video; no held-out video.
- Pipeline: crop below ROI_TOP=0.25 → segment → area gate (1–15% of frame) + ellipse-fit ≥ 0.85 → morphological-opening fallback (11/21/31) to sever text/arrow attachments → constant-velocity Kalman + Hungarian matching (gate 0.12×width) → MIN_HITS 3, MAX_MISSED 3.0 s → reset on camera cuts (32-bin histogram correlation < 0.6); IDs never reused.
- Result (1440 frames, 24 fps): 14 cut frames, 13 track IDs, 1 suspect ID switch (ID 2 → ID 10) flagged for visual check. Output: tracked_eggs.mp4 (H.264).
- Known limitation: the checkered shirt produces false detections/tracks (visible in preview sheets and the output video).

AI assistance is disclosed in docs/AI_USAGE_LOG.md.
