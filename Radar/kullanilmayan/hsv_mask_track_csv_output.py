#!/usr/bin/env python3
"""
HSV mask + object tracking demo with CSV export.

Usage examples:
  # 1) Webcam (default camera index 0), profile at /mnt/data/profile.json, write CSV to /mnt/data/track_points_sample.csv
  python hsv_mask_track.py --config /mnt/data/profile.json --csv-out /mnt/data/track_points_sample.csv

  # 2) With a specific video file
  python hsv_mask_track.py --video /path/to/video.mp4 --csv-out track.csv

Controls:
  q : quit
  r : re-initialize tracker from current mask
"""
import argparse
import json
from collections import deque
import csv

import cv2
import numpy as np


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expecting the schema the user provided
    p = data.get("params", {})
    hsv_low  = np.array(p.get("hsv_low",  [0, 0, 0]), dtype=np.uint8)
    hsv_high = np.array(p.get("hsv_high", [179, 255, 255]), dtype=np.uint8)

    blur_ksize = int(p.get("blur", 0)) or 0
    if blur_ksize % 2 == 0 and blur_ksize > 0:
        blur_ksize += 1  # GaussianBlur requires odd ksize

    canny = p.get("canny", [0, 0])
    c1, c2 = int(canny[0]) if len(canny) > 0 else 0, int(canny[1]) if len(canny) > 1 else 0

    kernel_size = int(p.get("kernel", 0)) or 0
    if kernel_size < 1:
        kernel_size = 1

    iters = int(p.get("iters", 1)) or 1
    use_close = bool(p.get("use_close", False))

    return {
        "hsv_low": hsv_low,
        "hsv_high": hsv_high,
        "blur": blur_ksize,
        "canny1": c1,
        "canny2": c2,
        "kernel": kernel_size,
        "iters": iters,
        "use_close": use_close,
    }


def make_mask(frame_bgr, cfg):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    if cfg["blur"] > 0:
        hsv = cv2.GaussianBlur(hsv, (cfg["blur"], cfg["blur"]), 0)

    mask = cv2.inRange(hsv, cfg["hsv_low"], cfg["hsv_high"])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["kernel"], cfg["kernel"]))
    if cfg["use_close"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg["iters"])
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg["iters"])

    # Optional edge emphasis via Canny (if thresholds are set > 0)
    if cfg["canny1"] > 0 and cfg["canny2"] > 0:
        edges = cv2.Canny(mask, cfg["canny1"], cfg["canny2"])
        # combine edges to reinforce contours (not strictly necessary)
        mask = cv2.bitwise_or(mask, edges)

    return mask


def largest_contour_bbox(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def create_tracker():
    # Try to construct a robust tracker; fall back gracefully based on OpenCV build
    tracker = None
    # Newer OpenCV
    if hasattr(cv2, "TrackerCSRT_create"):
        tracker = cv2.TrackerCSRT_create()
    # Legacy module
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        tracker = cv2.legacy.TrackerCSRT_create()
    # KCF as backup
    elif hasattr(cv2, "TrackerKCF_create"):
        tracker = cv2.TrackerKCF_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        tracker = cv2.legacy.TrackerKCF_create()
    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to input video. If omitted, uses webcam 0.")
    parser.add_argument("--config", type=str, default="profile.json", help="Path to HSV profile JSON.")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area to consider for init.")
    parser.add_argument("--csv-out", type=str, default="track_points_sample.csv",
                        help="CSV output for tracked center points (u,v).")
    args = parser.parse_args()

    cfg = load_profile(args.config)

    cap = cv2.VideoCapture("k1.mp4")
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    tracker = None
    has_tracker = False
    trail = deque(maxlen=64)  # points trail for visualization

    # Prepare CSV output
    csv_file = open(args.csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["u", "v"])  # header

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mask = make_mask(frame, cfg)

            wrote_point = False
            # Initialize / re-initialize tracker if needed
            if not has_tracker:
                bbox = largest_contour_bbox(mask, min_area=args.min_area)
                if bbox is not None:
                    tracker = create_tracker()
                    if tracker is not None:
                        has_tracker = tracker.init(frame, bbox)
                    else:
                        has_tracker = False
                # Draw bbox if found even before tracker
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cx, cy = x + w // 2, y + h // 2
                    trail.append((cx, cy))
                    csv_writer.writerow([cx, cy])
                    wrote_point = True

            else:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cx, cy = x + w // 2, y + h // 2
                    trail.append((cx, cy))
                    csv_writer.writerow([cx, cy])
                    wrote_point = True
                else:
                    has_tracker = False  # lost -> try reinit next loop

            # Draw trail
            for i in range(1, len(trail)):
                if trail[i - 1] is None or trail[i] is None:
                    continue
                cv2.line(frame, trail[i - 1], trail[i], (255, 255, 255), 2)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # force re-init next loop
                has_tracker = False
                trail.clear()

            # flush row to disk if we wrote one this frame
            if wrote_point:
                csv_file.flush()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()


if __name__ == "__main__":
    main()
