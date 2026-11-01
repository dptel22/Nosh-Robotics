# Nosh Robotics - Computer Vision Internship Assignment

Egg segmentation and video tracking pipeline (Tasks 1–4).

---

## How to run Task 4 (Video Tracking)

### Prerequisites
Install the required dependencies using Python 3.12:
```powershell
py -3.12 -m pip install -r requirements_task4.txt
```

### 1. Preview Mode (Inspect Before/After Tuning Sheets)
Generates 12 sample frames comparing frozen config segmentation vs. table ROI video segmentation with contours:
```powershell
py -3.12 task4_track.py --video video/Raw-and-Boiled-SnapYT.App.mp4 --config outputs/final_config.json --out output/task4 --preview
```
Output files:
- `output/task4/preview_before.png`
- `output/task4/preview_after.png`

### 2. Full Tracking Run
Executes egg segmentation, table ROI cropping, morphological attachment cutting, Kalman filtering, Hungarian data association, and camera cut detection across the complete video:
```powershell
py -3.12 task4_track.py --video video/Raw-and-Boiled-SnapYT.App.mp4 --config outputs/final_config.json --out output/task4
```

### Output Deliverables (in `output/task4/`)
- `tracked_eggs.mp4`: Annotated video with bounding contours, unique ID tags, and trajectory trails.
- `frame_log.csv`: Per-frame log (`frame`, `time_s`, `detections`, `tracked`).
- `events.csv`: Lifecycle events (`what`, `id`, `frame`, `time_s`, `position`).
- `plot_eggs_per_frame.png`: Plot of eggs tracked per frame over time with camera cuts marked as red dashed lines.
- `plot_trajectories.png`: X-position trajectories over time per track ID.
- `settings_used.json`: Complete dictionary of all constants and configuration parameters used.
- `summary.txt`: Video specifications, camera cut frames, track ID lifecycles, and suspect ID switch audit.
