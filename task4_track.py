"""How this works:
1. Loads frozen parameters from final_config.json and merges video-specific overrides.
2. In each frame, ignores the upper ROI, detects eggs with seg(), and uses morphological opening to sever thin attachments.
3. Tracks eggs across frames using a constant-velocity Kalman filter and Hungarian distance matching.
4. Detects camera cuts via frame histogram correlation and resets active tracks to prevent identity jumps across scenes.
5. Renders contours, IDs, and trajectory trails, writing the annotated video and diagnostic plots/logs.
"""

import argparse
import collections
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import egg_segmenter

# ---------------------------------------------------------------------------
# All tunable constants in ONE dictionary (starting guesses)
# ---------------------------------------------------------------------------
CONFIG_DEFAULTS = {
    # Starting guess: video overrides to find white eggs by HSV shape without dropping merged blobs
    "VIDEO_OVERRIDES": {
        "mode": "hsv",
        "s_max": 70,
        "v_min": 170,
        "morph": True,
        "select": "shape",
        "refine": None,
        "min_area": 0.01,
        "max_area": 0.5,
        "min_solid": 0.0,  # Do not drop odd-shaped merged blobs before detect() can cut them apart
    },
    # Starting guess: top fraction of frame to ignore (chef shirt, kitchen background)
    "ROI_TOP": 0.25,
    # Starting guess: plausible egg area range as fraction of total frame area
    "AREA_RANGE": (0.01, 0.15),
    # Starting guess: blob_area / fitted_ellipse_area >= 0.85 indicates an egg shape
    "MIN_ELLIPSE_FIT": 0.85,
    # Starting guess: kernel sizes for morphological opening to cut thin arrows and text attachments
    "OPENING_SIZES": (11, 21, 31),
    # Starting guess: maximum matching gate distance as fraction of frame width
    "MAX_DIST_FRACTION": 0.12,
    # Starting guess: consecutive detection matches before assigning an official ID
    "MIN_HITS": 3,
    # Starting guess: duration in seconds to keep predicting an unseen track before death
    "MAX_MISSED_SECONDS": 3.0,
    # Starting guess: 32-bin gray histogram correlation threshold below which a camera cut is triggered
    "CUT_THRESHOLD": 0.6,
    # Starting guess: number of past positions retained in the drawing trail
    "TRAIL_LENGTH": 40,
}

# Distinct BGR color palette for drawing tracked eggs
ID_COLORS = [
    (0, 255, 0),    # Bright Green
    (255, 128, 0),  # Sky Blue / Cyan
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (200, 200, 0),  # Teal
    (147, 20, 255), # Deep Pink
    (50, 205, 50),  # Lime Green
    (238, 130, 238),# Violet
    (0, 69, 255),   # Red-Orange
]


def get_color(track_id):
    """Return a deterministic BGR color for a given track ID."""
    return ID_COLORS[(track_id - 1) % len(ID_COLORS)]


# ---------------------------------------------------------------------------
# Step 1: Configuration Resolution
# ---------------------------------------------------------------------------
def find_final_config(cli_config=None):
    """Locate final_config.json from CLI path or newest match in outputs directory."""
    if cli_config:
        p = Path(cli_config)
        if p.exists():
            return p
        print(f"Provided config path '{cli_config}' does not exist.")

    # Search outputs directory
    matches = list(Path("outputs").rglob("final_config.json"))
    if not matches:
        matches = list(Path("output").rglob("final_config.json"))

    if not matches:
        print("Error: final_config.json not found in outputs/ or output/.")
        sys.exit(1)

    # Pick the newest config file if multiple exist
    newest = max(matches, key=lambda p: p.stat().st_mtime)
    print(f"Using config file: {newest}")
    return newest


def load_configurations(config_path, overrides):
    """Merge frozen final_config with segmenter defaults and video overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    frozen_base = raw.get("config", raw)
    cfg_frozen = {**egg_segmenter.DEFAULTS, **frozen_base}
    cfg_video = {**cfg_frozen, **overrides}
    return cfg_frozen, cfg_video


# ---------------------------------------------------------------------------
# Step 2: Detection and Thin Attachment Cutting
# ---------------------------------------------------------------------------
def ellipse_fit_ratio(contour):
    """Ratio of contour area to its fitted ellipse area."""
    if len(contour) < 5:
        return 0.0
    area = cv2.contourArea(contour)
    if area < 1.0:
        return 0.0
    (cx, cy), (ma, ma_major), _ = cv2.fitEllipse(contour)
    ellipse_area = math.pi * ma * ma_major / 4.0
    return area / ellipse_area if ellipse_area > 0 else 0.0


def try_cut_attachments(blob, frame_area, area_range, min_ellipse_fit, opening_sizes):
    """Cut thin text/arrow attachments using morphological opening and retain egg pieces."""
    for k in opening_sizes:
        opened = cv2.morphologyEx(blob, cv2.MORPH_OPEN, egg_segmenter.kernel(k))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        pieces = []
        for i in range(1, count):
            p_area = stats[i, cv2.CC_STAT_AREA]
            if area_range[0] <= p_area / frame_area <= area_range[1]:
                piece_mask = (labels == i).astype(np.uint8)
                cnt = egg_segmenter.largest_contour(piece_mask)
                if ellipse_fit_ratio(cnt) >= min_ellipse_fit:
                    pieces.append((piece_mask, p_area, cnt))
        if pieces:
            return pieces  # Successfully separated egg from attachments
    return []


def detect(frame, cfg_video, roi_top, area_range, min_ellipse_fit, opening_sizes):
    """Detect egg candidates in the table region below ROI_TOP and return centroid, area, contour."""
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    y_offset = int(h * roi_top)
    table_crop = frame[y_offset:, :]

    # Run frozen segmenter on table crop
    mask = egg_segmenter.seg(table_crop, cfg_video)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    detections = []
    for k in range(1, count):
        area = stats[k, cv2.CC_STAT_AREA]
        area_frac = area / frame_area
        if area_frac < area_range[0]:
            continue

        blob = (labels == k).astype(np.uint8)
        cnt = egg_segmenter.largest_contour(blob)
        fit = ellipse_fit_ratio(cnt)

        # Direct egg match
        if fit >= min_ellipse_fit and area_frac <= area_range[1]:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"]) + y_offset
                cnt_shifted = cnt + np.array([[[0, y_offset]]])
                detections.append({"center": (cx, cy), "area": area, "contour": cnt_shifted})
        else:
            # Try cutting thin attachments
            pieces = try_cut_attachments(blob, frame_area, area_range, min_ellipse_fit, opening_sizes)
            for p_mask, p_area, p_cnt in pieces:
                M = cv2.moments(p_cnt)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"]) + y_offset
                    p_cnt_shifted = p_cnt + np.array([[[0, y_offset]]])
                    detections.append({"center": (cx, cy), "area": p_area, "contour": p_cnt_shifted})

    return detections


# ---------------------------------------------------------------------------
# Step 3: Tracker Classes (Kalman, Track, Tracker)
# ---------------------------------------------------------------------------
class Kalman:
    """Constant-velocity 2D Kalman filter: state [x, y, vx, vy]."""

    def __init__(self, init_center):
        # State: x, y, vx, vy
        self.x = np.array([init_center[0], init_center[1], 0.0, 0.0], dtype=np.float64)
        # State transition matrix
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        # Measurement matrix (measures x and y only)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        # Error covariance
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        # Process noise covariance
        self.Q = np.diag([2.0, 2.0, 4.0, 4.0])
        # Measurement noise covariance
        self.R = np.diag([4.0, 4.0])

    def predict(self):
        """Advance state by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, center):
        """Incorporate new position measurement."""
        z = np.array([center[0], center[1]], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x[0], self.x[1]


class Track:
    """Represents a single egg trajectory."""

    def __init__(self, detection, trail_length):
        self.track_id = None  # None until MIN_HITS matches
        self.kalman = Kalman(detection["center"])
        self.center = detection["center"]
        self.contour = detection["contour"]
        self.area = detection["area"]
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.trail = collections.deque(maxlen=trail_length)
        self.trail.append(self.center)
        self.first_frame = None
        self.last_frame = None
        self.frames_seen = 0
        self.history = []  # List of (time_s, x_pos)


class Tracker:
    """Manages active egg tracks, Hungarian matching, and lifecycle events."""

    def __init__(self, max_dist, min_hits, max_missed_frames, trail_length):
        self.max_dist = max_dist
        self.min_hits = min_hits
        self.max_missed_frames = max_missed_frames
        self.trail_length = trail_length
        self.tracks = []
        self.all_confirmed_tracks = []
        self.next_id = 1
        self.events = []  # List of dicts: what, id, frame, time_s, position

    def predict(self):
        """Predict expected positions for all active tracks."""
        for t in self.tracks:
            pred_x, pred_y = t.kalman.predict()
            t.center = (pred_x, pred_y)
            t.age += 1

    def update(self, detections, frame_idx, time_s):
        """Associate detections with tracks via Hungarian matching."""
        if not self.tracks:
            for d in detections:
                new_track = Track(d, self.trail_length)
                self.tracks.append(new_track)
            return

        if not detections:
            for t in self.tracks:
                t.time_since_update += 1
            self.prune_tracks(frame_idx, time_s)
            return

        # Cost matrix: Euclidean distance between track predictions and detections
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        dist_matrix = np.zeros((num_tracks, num_dets), dtype=np.float64)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                dx = t.center[0] - d["center"][0]
                dy = t.center[1] - d["center"][1]
                dist_matrix[i, j] = math.hypot(dx, dy)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= self.max_dist:
                matched_tracks.add(r)
                matched_dets.add(c)
                t = self.tracks[r]
                d = detections[c]
                new_x, new_y = t.kalman.update(d["center"])
                t.center = (new_x, new_y)
                t.contour = d["contour"]
                t.area = d["area"]
                t.trail.append(t.center)
                t.hits += 1
                t.time_since_update = 0

                # Confirm track after MIN_HITS
                if t.hits >= self.min_hits and t.track_id is None:
                    t.track_id = self.next_id
                    self.next_id += 1
                    t.first_frame = frame_idx
                    self.all_confirmed_tracks.append(t)
                    self.events.append({
                        "what": "born",
                        "id": t.track_id,
                        "frame": frame_idx,
                        "time_s": round(time_s, 2),
                        "position": f"({int(t.center[0])}, {int(t.center[1])})"
                    })

                if t.track_id is not None:
                    t.last_frame = frame_idx
                    t.frames_seen += 1
                    t.history.append((time_s, t.center[0]))

        # Increment missed frames for unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.time_since_update += 1

        # Spawn new candidate tracks for unmatched detections
        for j, d in enumerate(detections):
            if j not in matched_dets:
                new_track = Track(d, self.trail_length)
                self.tracks.append(new_track)

        self.prune_tracks(frame_idx, time_s)

    def prune_tracks(self, frame_idx, time_s):
        """Remove dead tracks: unconfirmed after 1 miss, confirmed after max_missed_frames."""
        surviving = []
        for t in self.tracks:
            if t.track_id is None:
                # Unconfirmed candidate dies immediately on 1 missed frame
                if t.time_since_update <= 0:
                    surviving.append(t)
            else:
                # Confirmed track dies after max_missed_frames
                if t.time_since_update <= self.max_missed_frames:
                    surviving.append(t)
                else:
                    self.events.append({
                        "what": "lost",
                        "id": t.track_id,
                        "frame": frame_idx,
                        "time_s": round(time_s, 2),
                        "position": f"({int(t.center[0])}, {int(t.center[1])})"
                    })
        self.tracks = surviving

    def reset(self, frame_idx, time_s):
        """Reset all active tracks upon a camera cut; log lost events for confirmed tracks."""
        for t in self.tracks:
            if t.track_id is not None:
                self.events.append({
                    "what": "lost",
                    "id": t.track_id,
                    "frame": frame_idx,
                    "time_s": round(time_s, 2),
                    "position": f"({int(t.center[0])}, {int(t.center[1])})"
                })
        self.tracks = []


# ---------------------------------------------------------------------------
# Step 4: Camera Cut Detection
# ---------------------------------------------------------------------------
def compute_small_gray_hist(frame):
    """Compute normalized 32-bin grayscale histogram on frame downscaled to 160x90."""
    small = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def is_camera_cut(prev_hist, curr_hist, threshold):
    """Check if correlation between consecutive grayscale histograms drops below threshold."""
    if prev_hist is None or curr_hist is None:
        return False
    corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return corr < threshold


# ---------------------------------------------------------------------------
# Step 5: Rendering and Video Output
# ---------------------------------------------------------------------------
def draw_tracks_on_frame(frame, tracks, frame_idx, time_s):
    """Draw contour, label, and trail ONLY for confirmed tracks seen in the CURRENT frame."""
    annotated = frame.copy()
    visible_count = 0

    for t in tracks:
        # Only draw confirmed tracks seen in the CURRENT frame
        if t.track_id is not None and t.time_since_update == 0:
            visible_count += 1
            color = get_color(t.track_id)

            # Draw contour outline
            if t.contour is not None:
                cv2.drawContours(annotated, [t.contour], -1, color, 2)

            # Draw trail
            pts = [np.array(p, dtype=np.int32) for p in t.trail]
            if len(pts) > 1:
                cv2.polylines(annotated, [np.array(pts)], isClosed=False, color=color, thickness=2)

            # Draw text label
            cx, cy = int(t.center[0]), int(t.center[1])
            label = f"egg {t.track_id}"
            cv2.putText(annotated, label, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Frame and time caption
    banner = f"Frame: {frame_idx:04d} | Time: {time_s:05.2f}s | Tracked: {visible_count}"
    cv2.putText(annotated, banner, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def convert_to_h264(input_video_path, output_video_path):
    """Transcode video to H.264 (libx264, yuv420p) for Google Drive compatibility."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = None

    if not ffmpeg_exe:
        print("\nWARNING: ffmpeg not found and imageio-ffmpeg not installed.")
        print(f"Keeping OpenCV mp4v video at: {input_video_path}")
        print("To enable H.264 transcoding for Google Drive, run: py -3.12 -m pip install imageio-ffmpeg")
        return False

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(input_video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_video_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # Remove intermediate raw video after successful transcoding
        if Path(input_video_path).exists() and Path(output_video_path).exists():
            Path(input_video_path).unlink()
        return True
    except Exception as e:
        print(f"WARNING: Transcoding to H.264 failed ({e}). Keeping raw video.")
        return False


# ---------------------------------------------------------------------------
# Step 6: Preview Mode (--preview)
# ---------------------------------------------------------------------------
def run_preview(video_path, cfg_frozen, cfg_video, config_dict, out_dir):
    """Save preview_before.png (frozen config) and preview_after.png (video config + ROI line)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, 12, dtype=int)
    roi_top = config_dict["ROI_TOP"]

    frames_before = []
    frames_after = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        y_roi = int(h * roi_top)

        # Before: run frozen config unchanged on whole frame, draw mask in red
        mask_frozen = egg_segmenter.seg(frame, cfg_frozen)
        before_vis = frame.copy()
        before_vis[mask_frozen > 0] = (0, 0, 255)
        cv2.putText(before_vis, f"Frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frames_before.append(before_vis)

        # After: run detect() on table crop, draw contours + ROI line
        dets = detect(
            frame, cfg_video, roi_top,
            config_dict["AREA_RANGE"],
            config_dict["MIN_ELLIPSE_FIT"],
            config_dict["OPENING_SIZES"]
        )
        after_vis = frame.copy()
        # Draw ROI boundary line
        cv2.line(after_vis, (0, y_roi), (w, y_roi), (0, 255, 255), 2)
        cv2.putText(after_vis, f"ROI_TOP = {roi_top}", (20, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Draw detected egg contours
        for d in dets:
            cv2.drawContours(after_vis, [d["contour"]], -1, (0, 255, 0), 2)
            cx, cy = int(d["center"][0]), int(d["center"][1])
            cv2.circle(after_vis, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(after_vis, f"Frame {idx} ({len(dets)} eggs)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frames_after.append(after_vis)

    cap.release()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save 3x4 grids
    for name, img_list in [("preview_before.png", frames_before), ("preview_after.png", frames_after)]:
        rows = []
        for r in range(3):
            cols = [cv2.resize(img_list[r * 4 + c], (320, 180)) for c in range(4)]
            rows.append(np.hstack(cols))
        grid = np.vstack(rows)
        save_path = out_dir / name
        cv2.imwrite(str(save_path), grid)
        print(f"Saved preview sheet: {save_path}")

    print("Preview complete. Inspect preview_before.png and preview_after.png to verify settings.")


# ---------------------------------------------------------------------------
# Step 7: Diagnostic Plots and Suspect ID Switch Detection
# ---------------------------------------------------------------------------
def plot_results(frame_log_df, cuts, out_dir):
    """Plot eggs tracked per frame over time and save to plot_eggs_per_frame.png."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_log_df["time_s"], frame_log_df["tracked"], label="Tracked Eggs", color="blue")
    ax.scatter(frame_log_df["time_s"], frame_log_df["detections"], color="gray", alpha=0.3, s=10, label="Detections")

    for cut_frame in cuts:
        cut_time = frame_log_df.loc[frame_log_df["frame"] == cut_frame, "time_s"]
        if not cut_time.empty:
            ax.axvline(x=cut_time.iloc[0], color="red", linestyle="--", alpha=0.7)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Egg Count")
    ax.set_title("Tracked Eggs Over Time (Red Dashed Lines = Camera Cuts)")
    ax.legend()
    plt.tight_layout()
    plot_path = out_dir / "plot_eggs_per_frame.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")


def plot_trajectories(confirmed_tracks, out_dir):
    """Plot X coordinate of each ID over time to visualize continuous trajectories."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for t in confirmed_tracks:
        if t.history:
            times = [h[0] for h in t.history]
            x_coords = [h[1] for h in t.history]
            color_rgb = np.array(get_color(t.track_id)[::-1]) / 255.0
            ax.plot(times, x_coords, label=f"Egg {t.track_id}", color=color_rgb, linewidth=2)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("X Position (pixels)")
    ax.set_title("Egg X-Trajectory Over Time per Track ID")
    if len(confirmed_tracks) <= 15:
        ax.legend(loc="upper right")
    plt.tight_layout()
    traj_path = out_dir / "plot_trajectories.png"
    plt.savefig(str(traj_path), dpi=150)
    plt.close()
    print(f"Saved trajectory plot: {traj_path}")


def find_suspect_id_switches(events_df, max_dist, cuts_frames):
    """Find pairs where ID A is lost and ID B is born within 30 frames nearby, not at a camera cut."""
    suspects = []
    lost_events = events_df[events_df["what"] == "lost"].to_dict("records")
    born_events = events_df[events_df["what"] == "born"].to_dict("records")

    for l in lost_events:
        l_frame = l["frame"]
        l_id = l["id"]
        # Parse position
        try:
            lx, ly = [float(v) for v in str(l["position"]).strip("()").split(",")]
        except Exception:
            continue

        for b in born_events:
            b_frame = b["frame"]
            b_id = b["id"]
            if b_id == l_id:
                continue
            delta_frames = b_frame - l_frame
            if 0 <= delta_frames <= 30:
                # Check if camera cut occurred between l_frame and b_frame
                cut_between = any(l_frame <= cf <= b_frame for cf in cuts_frames)
                if not cut_between:
                    try:
                        bx, by = [float(v) for v in str(b["position"]).strip("()").split(",")]
                        dist = math.hypot(bx - lx, by - ly)
                        if dist <= 2.0 * max_dist:
                            suspects.append({
                                "lost_id": int(l_id),
                                "born_id": int(b_id),
                                "lost_frame": l_frame,
                                "born_frame": b_frame,
                                "delta_frames": delta_frames,
                                "distance_px": round(dist, 1)
                            })
                    except Exception:
                        pass
    return suspects


# ---------------------------------------------------------------------------
# Main Video Tracking Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 4: Egg Segmentation and Video Tracking")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", default=None, help="Path to final_config.json")
    parser.add_argument("--out", default="outputs/task4", help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Generate 12-frame preview sheet and stop")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = find_final_config(args.config)
    cfg_frozen, cfg_video = load_configurations(config_path, CONFIG_DEFAULTS["VIDEO_OVERRIDES"])

    # If --preview is requested, generate before/after sheets and stop
    if args.preview:
        run_preview(video_path, cfg_frozen, cfg_video, CONFIG_DEFAULTS, out_dir)
        return

    # Open video for full tracking
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loaded video: {video_path.name} | {width}x{height} @ {fps:.1f} fps | {total_frames} frames")

    # Initialize Tracker
    max_dist = CONFIG_DEFAULTS["MAX_DIST_FRACTION"] * width
    max_missed_frames = int(CONFIG_DEFAULTS["MAX_MISSED_SECONDS"] * fps)
    tracker = Tracker(
        max_dist=max_dist,
        min_hits=CONFIG_DEFAULTS["MIN_HITS"],
        max_missed_frames=max_missed_frames,
        trail_length=CONFIG_DEFAULTS["TRAIL_LENGTH"]
    )

    # Temporary raw output video path (mp4v codec)
    raw_video_path = out_dir / "raw_tracked.mp4"
    final_video_path = out_dir / "tracked_eggs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_video_path), fourcc, fps, (width, height))

    cuts_found = []
    prev_hist = None
    frame_logs = []
    frame_idx = 0

    print("Beginning video tracking...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_s = frame_idx / fps
        curr_hist = compute_small_gray_hist(frame)

        # Step 4: Camera cut check
        if is_camera_cut(prev_hist, curr_hist, CONFIG_DEFAULTS["CUT_THRESHOLD"]):
            cuts_found.append(frame_idx)
            print(f"  [Camera Cut] detected at frame {frame_idx:04d} ({time_s:.2f}s)")
            tracker.reset(frame_idx, time_s)

        prev_hist = curr_hist

        # Step 2: Egg detection on table ROI
        tracker.predict()
        detections = detect(
            frame, cfg_video,
            CONFIG_DEFAULTS["ROI_TOP"],
            CONFIG_DEFAULTS["AREA_RANGE"],
            CONFIG_DEFAULTS["MIN_ELLIPSE_FIT"],
            CONFIG_DEFAULTS["OPENING_SIZES"]
        )

        # Step 3: Tracker association
        tracker.update(detections, frame_idx, time_s)

        # Step 5: Draw annotations
        annotated_frame = draw_tracks_on_frame(frame, tracker.tracks, frame_idx, time_s)
        writer.write(annotated_frame)

        # Count active visible eggs
        visible_count = sum(1 for t in tracker.tracks if t.track_id is not None and t.time_since_update == 0)
        frame_logs.append({
            "frame": frame_idx,
            "time_s": round(time_s, 2),
            "detections": len(detections),
            "tracked": visible_count
        })

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx:04d}/{total_frames} ({time_s:5.1f}s) | Detections: {len(detections)} | Tracked: {visible_count}")

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - start_time
    print(f"Tracking complete: {frame_idx} frames processed in {elapsed:.1f}s ({frame_idx / max(elapsed, 0.001):.1f} fps).")

    # Step 5b: Transcode to H.264
    print("Transcoding video to H.264...")
    convert_to_h264(raw_video_path, final_video_path)

    # Save frame log and events CSVs
    frame_log_df = pd.DataFrame(frame_logs)
    frame_log_df.to_csv(out_dir / "frame_log.csv", index=False)

    events_df = pd.DataFrame(tracker.events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=["what", "id", "frame", "time_s", "position"])
    events_df.to_csv(out_dir / "events.csv", index=False)

    # Save diagnostic plots
    plot_results(frame_log_df, cuts_found, out_dir)
    plot_trajectories(tracker.all_confirmed_tracks, out_dir)

    # Save settings used
    settings_payload = {
        "config_path": str(config_path.resolve()),
        "constants": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG_DEFAULTS.items()
        }
    }
    with open(out_dir / "settings_used.json", "w", encoding="utf-8") as f:
        json.dump(settings_payload, f, indent=2)

    # Step 7: Summary report
    suspect_switches = find_suspect_id_switches(events_df, max_dist, cuts_found)
    total_ids = tracker.next_id - 1

    summary_lines = [
        "=" * 60,
        "TASK 4 TRACKING SUMMARY REPORT",
        "=" * 60,
        f"Input Video: {video_path.resolve()}",
        f"Resolution: {width} x {height} | FPS: {fps:.1f} | Total Frames: {frame_idx}",
        f"Camera Cuts Detected: {len(cuts_found)} (frames: {cuts_found})",
        f"Total Unique Track IDs Assigned: {total_ids}",
        "-" * 60,
        "TRACK ID LIFECYCLES:",
    ]

    for t in tracker.all_confirmed_tracks:
        first_s = (t.first_frame / fps) if t.first_frame is not None else 0.0
        last_s = (t.last_frame / fps) if t.last_frame is not None else first_s
        summary_lines.append(
            f"  Egg ID {t.track_id:02d}: first seen {first_s:5.2f}s, last seen {last_s:5.2f}s "
            f"({t.frames_seen} frames matched)"
        )

    summary_lines.append("-" * 60)
    summary_lines.append(f"POSSIBLE ID SWITCHES (SUSPECTS ONLY - CHECK BY EYE): {len(suspect_switches)}")
    if suspect_switches:
        for s in suspect_switches:
            summary_lines.append(
                f"  Suspect Switch: ID {s['lost_id']} lost -> ID {s['born_id']} born "
                f"within {s['delta_frames']} frames ({s['distance_px']}px apart)"
            )
    else:
        summary_lines.append("  None detected.")
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nAll deliverables written to: {out_dir.resolve()}")


if __name__ == "__main__":
    cv2.setRNGSeed(0)
    main()
