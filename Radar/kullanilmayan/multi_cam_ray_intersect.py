#!/usr/bin/env python3
import argparse
import json
import math
import csv
from typing import Optional, Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ----------------------- HSV mask + centroid -----------------------
def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = data.get("params", {})
    cfg = {
        "hsv_low": np.array(p.get("hsv_low", [0,0,0]), dtype=np.uint8),
        "hsv_high": np.array(p.get("hsv_high", [179,255,255]), dtype=np.uint8),
        "blur": int(p.get("blur", 0)) or 0,
        "canny": p.get("canny", [0,0]),
        "kernel": int(p.get("kernel", 3)) or 3,
        "iters": int(p.get("iters", 1)) or 1,
        "use_close": bool(p.get("use_close", False)),
    }
    if cfg["blur"] > 0 and cfg["blur"] % 2 == 0:
        cfg["blur"] += 1
    if cfg["kernel"] < 1:
        cfg["kernel"] = 1
    return cfg


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
    c1, c2 = 0, 0
    if isinstance(cfg["canny"], (list, tuple)) and len(cfg["canny"]) >= 2:
        c1, c2 = int(cfg["canny"][0]), int(cfg["canny"][1])
    if c1 > 0 and c2 > 0:
        mask = cv2.bitwise_or(mask, cv2.Canny(mask, c1, c2))
    return mask


def largest_contour_centroid(mask, min_area=300) -> Optional[Tuple[int,int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))


# ----------------------- Camera model -----------------------
def build_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0, 1.0]], float)


def backproject_dir(u, v, K_inv):
    uv1 = np.array([u, v, 1.0], float)
    d_cam = K_inv @ uv1
    d_cam /= np.linalg.norm(d_cam)
    return d_cam  # camera coords (z forward)


def look_at_rotation(cam_pos, target=np.array([0,0,0], float), up_world=np.array([0,0,1], float)):
    """
    Build world_from_cam rotation so that camera's +Z axis looks toward target.
    Convention: camera coords: +Z forward, +X right, +Y up.
    """
    z_cam_world = (target - cam_pos)
    z_cam_world = z_cam_world / np.linalg.norm(z_cam_world)

    x_cam_world = np.cross(z_cam_world, up_world)
    if np.linalg.norm(x_cam_world) < 1e-8:
        # camera is parallel to up vector; pick arbitrary right
        x_cam_world = np.array([1,0,0], float)
    x_cam_world = x_cam_world / np.linalg.norm(x_cam_world)

    y_cam_world = np.cross(x_cam_world, z_cam_world)
    y_cam_world = y_cam_world / np.linalg.norm(y_cam_world)

    # Columns are the world vectors of cam axes: [Xw, Yw, Zw]
    R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])
    return R_wc


# ----------------------- Multi-line intersection -----------------------
def intersect_rays_least_squares(origins: List[np.ndarray], dirs: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Given N rays (O_i, d_i), compute argmin_X sum_i ||(I - d_i d_i^T)(X - O_i)||^2.
    Closed-form: (sum A_i) X = sum A_i O_i, with A_i = (I - d_i d_i^T).
    Returns X and RMS distance to the rays.
    """
    I = np.eye(3)
    A = np.zeros((3,3), float)
    b = np.zeros(3, float)
    for O, d in zip(origins, dirs):
        d = d / np.linalg.norm(d)
        Ai = I - np.outer(d, d)
        A += Ai
        b += Ai @ O
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    # compute rms distance
    dists = []
    for O, d in zip(origins, dirs):
        diff = X - O
        dists.append(np.linalg.norm(np.cross(diff, d)))
    rms = float(np.sqrt(np.mean(np.square(dists)))) if len(dists) > 0 else float('nan')
    return X, rms


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=str, required=True, help="k1.mp4")
    ap.add_argument("--v2", type=str, required=True, help="k2.mp4")
    ap.add_argument("--v3", type=str, required=True, help="k3.mp4")
    ap.add_argument("--v4", type=str, required=True, help="k4.mp4")
    ap.add_argument("--config", type=str, default="profile.json")
    # Intrinsics (assume same for all cams by default; can override per-cam later if needed)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    # Geometry: square of side 2*b centered at origin at height H
    ap.add_argument("--b", type=float, default=2.0, help="Half side length of square (meters)")
    ap.add_argument("--H", type=float, default=2.0, help="Camera height (meters)")
    # Logging / plotting
    ap.add_argument("--out", type=str, default="intersections.csv")
    ap.add_argument("--plot", action="store_true", help="Show live 3D plot")
    ap.add_argument("--min_area", type=int, default=300, help="Min contour area")
    ap.add_argument("--rms_tol", type=float, default=0.15, help="Max RMS distance (m) to accept intersection")
    args = ap.parse_args()

    # Open caps
    caps = [cv2.VideoCapture(p) for p in (args.v1, args.v2, args.v3, args.v4)]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise SystemExit(f"Could not open video {i+1}")

    # Read one frame to set intrinsics
    ok, frame0 = caps[0].read()
    if not ok:
        raise SystemExit("Could not read first frame from v1.")
    Hpix, Wpix = frame0.shape[:2]
    fx = args.fx if args.fx is not None else 0.8 * Wpix
    fy = args.fy if args.fy is not None else 0.8 * Wpix
    cx = args.cx if args.cx is not None else Wpix / 2.0
    cy = args.cy if args.cy is not None else Hpix / 2.0
    K = build_intrinsics(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)

    cfg = load_profile(args.config)

    # Camera world positions (square corners at z=H)
    b = args.b
    Hm = args.H
    cam_positions = [
        np.array([+b, +b, Hm], float),
        np.array([-b, +b, Hm], float),
        np.array([-b, -b, Hm], float),
        np.array([+b, -b, Hm], float),
    ]
    target = np.array([0.0, 0.0, 0.0], float)
    R_wcs = [look_at_rotation(p, target=target) for p in cam_positions]  # world_from_cam

    # Plot setup
    if args.plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("4-Cam Ray Intersection")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_box_aspect([1,1,0.6])
        ax.view_init(elev=25, azim=45)
        # Draw ground
        s = 3.5
        gx = np.linspace(-s, s, 2); gy = np.linspace(-s, s, 2)
        GX, GY = np.meshgrid(gx, gy); GZ = np.zeros_like(GX)
        ground = ax.plot_surface(GX, GY, GZ, alpha=0.15, edgecolor='k')
        # Draw camera positions
        ax.scatter([p[0] for p in cam_positions], [p[1] for p in cam_positions], [p[2] for p in cam_positions], s=30)
        # Create ray line artists
        ray_lines = [ax.plot([0,0],[0,0],[0,0], linewidth=1)[0] for _ in range(4)]
        # Intersection point artist
        inter_scatter = ax.scatter([0],[0],[0], s=40)
        # Trace
        trace_X, trace_Y, trace_Z = [], [], []

    # Logging
    fout = open(args.out, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["frame","time_s","x_m","y_m","z_m","rms_m"])

    # Sync assumptions: same fps, same start. We'll step one frame per cap per loop.
    frame_idx = 0
    while True:
        frames = []
        times = []
        for cap in caps:
            ok, fr = cap.read()
            if not ok:
                frames = []
                break
            frames.append(fr)
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if not frames:
            break

        # Find centroids
        origins = []
        dirs = []
        have_all = True
        for i in range(4):
            mask = make_mask(frames[i], cfg)
            c = largest_contour_centroid(mask, min_area=args.min_area)
            if c is None:
                have_all = False
                break
            u, v = float(c[0]), float(c[1])
            d_cam = backproject_dir(u, v, K_inv)
            d_world = R_wcs[i] @ d_cam
            origins.append(cam_positions[i])
            dirs.append(d_world)

            # debug windows
            vis = frames[i].copy()
            cv2.circle(vis, (int(u), int(v)), 6, (0,255,0), -1)
            cv2.imshow(f"cam{i+1}", vis)

        if have_all:
            X, rms = intersect_rays_least_squares(origins, dirs)
            if rms <= args.rms_tol:
                t_mean = float(np.mean(times))
                writer.writerow([frame_idx, t_mean, X[0], X[1], X[2], rms])

                if args.plot:
                    # update rays (finite length)
                    L = 5.0
                    for i in range(4):
                        P_far = origins[i] + dirs[i] * L
                        ray_lines[i].set_data_3d([origins[i][0], P_far[0]],
                                                 [origins[i][1], P_far[1]],
                                                 [origins[i][2], P_far[2]])
                    # update intersection scatter
                    inter_scatter._offsets3d = ([X[0]], [X[1]], [X[2]])
                    # update trace
                    trace_X.append(X[0]); trace_Y.append(X[1]); trace_Z.append(X[2])
                    ax.plot(trace_X, trace_Y, trace_Z, linewidth=2)

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    # cleanup
    fout.close()
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    if args.plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
