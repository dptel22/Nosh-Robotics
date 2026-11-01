# FACT SHEET — every number the report uses, with its source

Rule: every value below was read from a file in `output\` or from a saved notebook cell output.
Nothing is invented. "Cell n" = Task_n.ipynb cell index (0-based, as saved in the .ipynb).
Note: the repo folder is `output\` (singular). Verified 2026-11-01.

## 1. Dataset

| Claim | Value | Source |
|---|---|---|
| Dataset | Roboflow export "egg.v3i.folder", project fire-bf7qe/egg-vzhtr, CC BY 4.0 | data/egg.v3i.folder/README.dataset.txt, README.roboflow.txt |
| Export date | 2025-05-23 7:27 AM GMT | README.roboflow.txt line 1 |
| Pre-processing | Auto-orientation, resize to 640x640 (stretch) | README.roboflow.txt |
| Total images | 160 | Task_1.ipynb cell 10 output ("Total Images Loaded: 160"); file counts below |
| Roboflow folder counts | train 112 (110 in `egg/` + 2 in `Unlabeled/`), valid 32, test 16 = 70/20/10 % | file counts in data/egg.v3i.folder/*; mask dirs in output/task1_outputs/ |
| Filename pattern | `not_damaged_<n>_jpg.rf.<hash>.jpg` (Roboflow augmentation suffix) | data/egg.v3i.folder filenames |
| Two `Unlabeled` images | not_damaged_135, not_damaged_145 (train) | output/task1_outputs/sam_report.csv rows; folder listing |
| Base-name duplicates in the 160 | 0 groups share a base name before `_jpg.rf.` | computed from filenames, 2026-11-01 (reportable as a check I ran) |

## 2. Task 1 — SAM 3 pseudo ground truth

| Claim | Value | Source |
|---|---|---|
| Model | SAM 3, github.com/facebookresearch/sam3, HF checkpoint `sam3.pt` 3.45 GB | Task_1.ipynb cells 2, 6, 12 (code + download output) |
| Environment | Google Colab, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA (T4), NumPy 1.26.4 | Task_1.ipynb cell 6 output |
| Text prompt (main pass) | "egg" | Task_1.ipynb cell 14 (`set_text_prompt(prompt="egg")`) |
| Confidence threshold (main) | 0.30 | Task_1.ipynb cell 12 (`Sam3Processor(..., confidence_threshold=0.30)`) |
| Mask NMS | min_area 0.002, max_area 0.98, IoU threshold 0.70, GPU-vectorized | Task_1.ipynb cell 14 (`select_masks_gpu`) |
| Instance painting | reverse-score-order so high-confidence masks overwrite overlaps | Task_1.ipynb cell 18 comment |
| Flags written | NO_DETECTION / MULTI_EGG (>1 kept) / LOW_SCORE (<0.5) / VERY_LARGE_MASK (area>0.90) | Task_1.ipynb cell 18 (flag block) |
| First pass NO_DETECTION | 10 images | Task_1.ipynb cell 19 output ("Found 10 NO_DETECTION images") |
| Recovery patch | fallback prompts ["egg","boiled egg","brown egg","white egg","oval object"], threshold 0.12, best mask only | Task_1.ipynb cell 19 (code) |
| Recovered | 7/10: 82 (0.285, "egg"), 135 (0.162, "egg"), 145 (0.146, "egg"), 117 (0.185, "white egg"), 124 (0.133, "white egg"), 44 (0.164, "egg"), 69 (0.174, "egg") | Task_1.ipynb cell 19 output; sam_report.csv rows |
| Still empty after patch | 3: not_damaged_42, not_damaged_77, not_damaged_116 (all train folder) | Task_1.ipynb cell 19 output (7/10 recovered) + sam_report.csv NO_DETECTION rows + pixel check of sam_masks & union_masks (2026-11-01) |
| QA summary | 160 images, avg 3.71 s/img; CLEAN_SINGLE_EGG 126 (78.8%), MULTI_EGG 22 (13.8%), LOW_SCORE_RECOVERY 7 (4.4%), LOW_SCORE 3 (1.9%), NO_DETECTION 3 (1.9%), VERY_LARGE_MASK 1 (0.6%) | Task_1.ipynb cell 21 output (flag-token counts; LOW_SCORE 3 = 1 LOW_SCORE + 2 MULTI_EGG\|LOW_SCORE rows; MULTI_EGG 22 = 20 + 2) |
| sam_report.csv flag rows (raw) | '' 126, MULTI_EGG 20, LOW_SCORE_RECOVERY 7, MULTI_EGG\|LOW_SCORE 2, NO_DETECTION 3, LOW_SCORE 1, VERY_LARGE_MASK 1 | output/task1_outputs/sam_report.csv (160 rows) |
| n_eggs distribution | 0→3 imgs, 1→135, 2→14, 3→8 | sam_report.csv `n_eggs` column |
| VERY_LARGE_MASK image | not_damaged_48 (score 0.914, area>0.90) | sam_report.csv row |
| LOW_SCORE image | not_damaged_123 (0.391) | sam_report.csv row |
| Contact sheet | built+displayed in cell 21, never saved to disk; extracted to output/report/task1_contact_sheet.png from cell 21 stored output | Task_1.ipynb cell 21; extraction 2026-11-01 |

## 3. Task 2 — split (split.json)

| Claim | Value | Source |
|---|---|---|
| Grouping rule | pHash Hamming distance ≤ 10 OR identical base name (`.rf.<hash>` stripped) → connected components | Task_2.ipynb cell 4 (code); split.json `thr:10` |
| Splitter | sklearn GroupShuffleSplit, test_size=0.2, random_state=42 | Task_2.ipynb cell 4; split.json `seed:42` |
| Groups at each threshold | {0:160, 4:159, 8:157, 10:156, 12:156, 16:144} | Task_2.ipynb cell 4 output |
| Groups at THR=10 | 156, largest group size 2 | Task_2.ipynb cell 4 output |
| Split result | dev 127 / eval 33 (3.85:1); no group crosses splits (asserted in code) | Task_2.ipynb cell 4 output + split.json |
| Closest dev-vs-eval pHash distance | 16 | Task_2.ipynb cell 4 output |
| Dev composition by Roboflow folder | train 88, valid 26, test 13 | computed from split.json + folder walk (2026-11-01) |
| Eval composition by Roboflow folder | train 24, valid 6, test 3 | same |
| Audited pHash pairs (docs) | 78↔27 d=2, 51↔45 d=6, 60↔8 d=6 (true dups); 1↔7 d=10 (false merge, kept grouped) | docs/TASK_2_EXPERIMENT_LOG.md §1; docs/AI_USAGE_LOG.md Incident 9 (docs only — not in machine-readable outputs) |
| Roboflow folders NOT used for the split | 112/32/16 is not 4:1; folder membership mixes burst duplicates across folders | file counts; Task_2 cell 4 rationale in docs |

## 4. Task 2 — trials (experiments.csv + notebook outputs)

| Trial | Change (cfg delta) | why (CSV column) | dev mIoU | median | Dice | P | R | ms/img | worst5 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | {} (baseline: gauss+Otsu gray+largest) | "baseline: gray + Gaussian + Otsu + largest blob (zero tuning)" | 0.5558 | 0.5809 | 0.665 | 0.6734 | 0.735 | 76 | 10;39;42;67;77 | KEPT (frozen) |
| T1 | mode=hsv | "white egg = high V, LOW saturation; should reject warm skin and dark background" | 0.5588 | 0.5661 | 0.6698 | 0.7149 | 0.7207 | 52 | 10;24;28;39;42 | discarded (reconstructed reason — see decisions.md) |
| T2 | select=score | "pick the most egg-like blob (convex, ellipse-like, central, big); wrong blob won in 108, 28, 39" | 0.5462 | 0.5752 | 0.6517 | 0.6635 | 0.7077 | 54 | 12;10;11;28;39 | discarded (worse mIoU) |
| T3 | mode=grabcut | "no colour rule: eggs here are white, brown, on any background; assume big and central" | 0.5402 | 0.5768 | 0.6469 | 0.6793 | 0.6628 | 8670 | 10;13;42;56;67 | discarded (worse mIoU + 114x slower) |

Sources: T0–T2 from output/egg_task2_out/content/egg_task2_out/experiments.csv (verified identical to the out/ copy and byte-equal copies of final_config/split). T3 ONLY from Task_2.ipynb cell 14 saved output — NOT present in experiments.csv (contradiction logged; Dhruv informed 2026-11-01, no objection recorded → report cites notebook output for T3 and says the CSV is missing the row).
Note: experiments.csv `decision` column is empty for all rows (notebook comment says "fill 'decision' BY HAND"; never filled).
Note: no per-image "T1 vs T0 better on N images" comparison line exists in any saved notebook output (checked 2026-11-01).
Note: dev mIoU covers all 127 dev images; the 2 empty-key images score IoU 0 (they appear in worst5: 42 in T0/T1/T3 worst5).

## 5. Final frozen config (final_config.json — output root + egg_task2_out copies, all identical)

```
denoise=gauss, mode=otsu_gray, s_max=65, v_min=150, l_min=140, ab_tol=22, skin=False,
morph=False, k_open=5, k_close=15, select=largest, min_area=0.005, max_area=0.98,
min_solid=0.9, refine=None, gc_margin=0.08, gc_iters=4
```
Source: output/final_config.json; Task_2.ipynb cell 15 output ("frozen: {...}").
Facts about it:
- select="largest" ignores min_area, max_area, min_solid (code path: argmax over ALL component areas) — Task_2.ipynb cell 6 / Task_3.ipynb cell 17 / egg_segmenter.py `pick_blobs`.
- Frozen config uses NO GrabCut (refine=None; mode=otsu_gray). Task 2 trials ran GrabCut unseeded (no cv2.setRNGSeed anywhere in Task_2.ipynb) → T3's numbers not exactly reproducible. Task_3.ipynb cell 19 and egg_segmenter.py add `cv2.setRNGSeed(0)`; task4_track.py seeds at main().
- pred_masks for all 160 images generated with the frozen config (Task_2 cell 15).

## 6. Task 3 — evaluation (frozen config vs SAM masks)

Protocol: same split.json as Task 2; SAM masks = answer key; every image scored exactly once
(Task_3.ipynb cell 29 comment: "Score every image once with the frozen config. Eval is scored here and nowhere else.");
empty-key rows kept in CSVs but excluded from averages (`dv_ok`/`ev_ok`).
Source: output/task3_outputs/content/task3_outputs/*.csv; Task_3.ipynb cells 29–43 outputs.

| Metric (mean / median) | dev n=125 (excl. 2 empty) | eval n=32 (excl. 1 empty) |
|---|---|---|
| IoU | 0.565 / 0.589 | 0.615 / 0.677 |
| Dice | 0.676 / 0.741 | 0.729 / 0.807 |
| precision | 0.684 / 0.776 | 0.793 / 0.970 |
| recall | 0.747 / 0.825 | 0.746 / 0.797 |
| boundary F1 (BF1, 3 px) | 0.230 / 0.160 | 0.240 / 0.186 |
| HD95 (px) | 207.280 / 199.312 | 188.202 / 176.840 |

(Per-split std/min in summary.csv; dev IoU std 0.274 min 0.0; eval std 0.246 min 0.046.)
- Full-split means incl. empty keys: dev 127 rows mean IoU 0.5558, eval 33 rows mean 0.5966 (computed from dev_metrics.csv / eval_metrics.csv, 2026-11-01). The 0.5558 exactly equals T0's logged dev_miou → same masks, same 127-image protocol.
- Bootstrap CI: eval mean IoU 0.615, 95% interval 0.531–0.692, n=32, 1000 resamples, seed 0 (Task_3 cell 31 output).
- dev − eval gap: −0.051 (eval higher) (Task_3 cell 31 output).
- Why eval > dev: dev is harder — multi-egg keys 27/127 dev vs 8/33 eval; IoU<0.1 images 15 dev (13 non-empty) vs 2 eval (1 non-empty) (computed from dev/eval_metrics.csv).
- Runtime: 30 ms/image (eval mean) (Task_3 cell 33 output).

Instance-level (Hungarian matching, tp at IoU≥0.5; key instances from SAM label IDs when present — touching eggs stay separate — else connected components ≥0.2% area):
| | dev (n=125) | eval (n=32) |
|---|---|---|
| precision | 0.624 | 0.812 |
| recall | 0.473 | 0.605 |
| F1 | 0.538 | 0.693 |
| count error (mean |n_pred−n_gt|) | 0.32 | 0.344 |
Source: Task_3.ipynb cell 33 output. n_pred=1 on all 32 usable eval images; n_gt: 24×1, 5×2, 3×3 (eval_usable_taxonomy.csv).
Excluded: 1 eval image (116) with empty key; the algorithm still painted something on it (Task_3 cell 33 output).

Error taxonomy (automatic `error_shape`, eval n=32): mixed 14, under_segmented 9, over_segmented 5, good 3, wrong_object 1 (Task_3 cell 39 output; eval_usable_taxonomy.csv).
Manual `cause` column: EMPTY (CAUSE dict never filled — Task_3 cell 41 code has only a commented example). All causes in the report are marked "(proposed – Dhruv to confirm)".
12 worst eval images with context (Task_3 cell 39 output): 117 IoU 0.05 wrong_object (egg_V 140 < bg_V 155); 51 0.16, 45 0.17 over_segmented; 23 0.22; 22 0.25 over_segmented; 147 0.36, 103 0.49, 35 0.51, 73 0.52, 114 0.56 under_segmented; 5 0.39; 62 0.52.
IoU≥0.85 on 3/32; IoU<0.1 on 1/32 (Task_3 cell 43 output).

## 7. Task 4 — video tracking

Source: output/task4/* (summary.txt, settings_used.json, frame_log.csv, events.csv); task4_track.py; TASK4_EXPLAINED.md.

| Claim | Value | Source |
|---|---|---|
| Input video | video/Raw-and-Boiled-SnapYT.App.mp4 | summary.txt line 4 |
| Resolution / fps / frames | 1280 x 720, 24.0 fps, 1440 frames (60 s) | summary.txt line 5 |
| Camera cuts | 14, frames [210, 218, 219, 220, 221, 1001, 1002, 1003, 1004, 1005, 1006, 1008, 1009, 1010] | summary.txt line 6 |
| Unique track IDs | 13 | summary.txt line 7 |
| Track lifecycles | ID1 9.29–36.92 s (563 f), ID2 9.29–30.42 s (507 f), ID3 21.04–21.17 (3), ID4 22.79–24.17 (19), ID5 23.25–27.04 (92), ID6 26.62–26.88 (5), ID7 28.42–28.54 (4), ID8 31.21–41.67 (252), ID9 31.75–35.62 (19), ID10 33.79–41.58 (188), ID11 35.62–40.46 (22), ID12 36.42–40.83 (81), ID13 36.75–41.62 (109) | summary.txt |
| Suspect ID switch | 1: ID 2 lost → ID 10 born within 8 frames, 189.4 px apart (events.csv: lost 2 @frame 803 (33.46 s) pos (868,659); born 10 @frame 811 (33.79 s) pos (765,500)) | summary.txt + events.csv |
| Detections per frame | mean 1.32, median 1, max 5; 658/1440 frames with 0 detections (intro/outro) | frame_log.csv (computed 2026-11-01) |
| Tracked per frame | mean 1.29, median 1, max 5 | frame_log.csv |
| Output video | output/task4/tracked_eggs.mp4 (H.264 transcode) | task4_track.py `final_video_path`; README |
| Drive link (folder) | https://drive.google.com/drive/folders/1hkSgiil8bvMaxICCLFTDfrn3z1tSsyrn?usp=sharing | Dhruv, 2026-11-01 (not opened by me) |

Segmenter overrides vs frozen config (task4_track.py CONFIG_DEFAULTS; identical in settings_used.json):
mode otsu_gray→hsv; s_max 65→70; v_min 150→170; morph False→True (k_open 5, k_close 15 unchanged);
select largest→shape; min_area 0.005→0.01; max_area 0.98→0.5; min_solid 0.9→0.0.
Unchanged: denoise=gauss, skin=False, refine=None.
The shipped s_max=70 / v_min=170 EQUAL the starting guesses recorded in settings_used.json (no further tuning round is recorded).
Video-only constants: ROI_TOP 0.25; AREA_RANGE (0.01, 0.15) of full frame; MIN_ELLIPSE_FIT 0.85; OPENING_SIZES (11,21,31);
MAX_DIST_FRACTION 0.12 (=153.6 px gate at 1280 px width); MIN_HITS 3; MAX_MISSED_SECONDS 3.0 (=72 frames at 24 fps);
CUT_THRESHOLD 0.6 (32-bin gray histogram correlation, frames downscaled to 160x90); TRAIL_LENGTH 40;
Kalman: constant-velocity [x,y,vx,vy], P=diag(10,10,100,100), Q=diag(2,2,4,4), R=diag(4,4) (task4_track.py Kalman class).
Tracker: Hungarian assignment (scipy linear_sum_assignment) with distance gate; unconfirmed candidate dies after 1 missed frame;
confirmed track dies after 3 s without detection; camera cut → reset() logs lost for all confirmed tracks, IDs never reused;
labels/trails drawn only for tracks matched in the current frame (time_since_update==0).
"lost" event positions are Kalman-predicted positions at prune time and can be far off-frame (e.g. ID 3 lost at (3284,−975)) — events.csv.

Shirt false positives (observed, from preview sheets + output video frames Dhruv screenshotted):
HSV rule (V≥170, S≤70) foregrounds the white checks of the chef's red/white checkered shirt; close(15) merges them;
select="shape" + min_solid=0.0 keeps every blob in the size window; the 11/21/31 opening fallback rounds shirt regions
until they pass MIN_ELLIPSE_FIT. preview_before.png shows the frozen Otsu config masking shirt/hands/graphics wholesale.
Per-ID identity (real egg / shirt / graphic) = [MISSING: Dhruv's ID table — placeholder left in instructions, table not pasted].

## 8. Figures available (all under output\ unless noted)

- Task 1: sam_overlays (160 jpg; e.g. clean: 101; recovery: 124, 82; VERY_LARGE_MASK: 48; MULTI_EGG: 103); output/report/task1_contact_sheet.png (extracted from Task_1.ipynb cell 21 output).
- Task 2: pred_masks (160 png) + originals — failure composites to be generated from these saved files (108, 28, 39, 24 named in CSV "why" column).
- Task 3: fig_quant.png, fig_worst.png, fig_taxonomy.png (zip, Nov 1).
- Task 4: preview_before.png, preview_after.png, plot_eggs_per_frame.png, plot_trajectories.png; tracked_eggs.mp4.

## 9. Known doc-vs-data contradictions (report follows the data; one sentence discloses the stale docs)

1. docs (TASK_2_EXPERIMENT_LOG.md, AI_USAGE_LOG.md Incident 10) say 135/145/124/82 had empty masks excluded from dev scoring. Data: those four were recovered (LOW_SCORE_RECOVERY); empties are 42/77/116; logged T0/T1 INCLUDE empties at IoU 0 (worst5 contains 42/77; 0.5558 = 127-image mean).
2. T3 (grabcut) ran per notebook cell 14 output but its row is absent from experiments.csv.
3. TASK_2_EXPERIMENT_LOG.md is stale: old T0 0.5334, "T2 KEPT" (frozen config is T0's), T3 "IN PROGRESS", T4 "Planned".
4. AI_USAGE_LOG.md Incident 11 says "trials T0–T6"; only T0–T3 exist.
5. README Task 4 commands reference `outputs\final_config.json`; real folder is `output\` (task4_track.py searches both). NOT mentioned in the PDF per Dhruv (2026-11-01); listed in the final handoff instead.
