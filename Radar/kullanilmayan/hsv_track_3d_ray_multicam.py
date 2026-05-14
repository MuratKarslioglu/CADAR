#!/usr/bin/env python3
import argparse
import json
import math
import sys

# --- Windows-friendly interactive backend ---
import matplotlib
for candidate in ("QtAgg", "Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(candidate, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import cv2


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


def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0],[0, c, -s],[0, s, c]], float)


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]], float)


def camera_rotation(yaw_deg, pitch_deg, roll_deg=0.0):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    return rot_z(yaw) @ rot_x(pitch) @ rot_z(roll)  # world_from_cam


def build_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1.0]], float)


def backproject_dir(u, v, K_inv):
    uv1 = np.array([u, v, 1.0], float)
    d_cam = K_inv @ uv1
    n = np.linalg.norm(d_cam)
    if n == 0:
        return np.array([0,0,1.0], float)
    d_cam /= n
    return d_cam


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


def largest_contour_centroid(mask, min_area=300):
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


def build_image_plane_mesh(W, H, K_inv, R_wc, z_cam=1.0, grid=20):
    uu = np.linspace(0, W-1, grid)
    vv = np.linspace(0, H-1, grid)
    U, V = np.meshgrid(uu, vv)
    Xw = np.zeros_like(U, float)
    Yw = np.zeros_like(U, float)
    Zw = np.zeros_like(U, float)
    for i in range(grid):
        for j in range(grid):
            u, v = U[i, j], V[i, j]
            d_cam = backproject_dir(u, v, K_inv)
            t = z_cam / (d_cam[2] if abs(d_cam[2]) > 1e-9 else 1.0)
            P_cam = d_cam * t
            Pw = R_wc @ P_cam
            Xw[i, j], Yw[i, j], Zw[i, j] = Pw[0], Pw[1], Pw[2]
    return U, V, Xw, Yw, Zw


def open_caps(video_list):
    caps = []
    for v in video_list:
        cap = cv2.VideoCapture(0 if v in ("0","webcam") else v)
        if not cap.isOpened():
            print(f"[WARN] Could not open source: {v}", file=sys.stderr)
            caps.append(None)
        else:
            caps.append(cap)
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["k1.mp4","k2.mp4","k3.mp4","k4.mp4"],
                    help="Up to 4 paths. Use 'webcam' or '0' for a webcam source.")
    ap.add_argument("--config", type=str, default="profile.json", help="HSV profile JSON")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--plane_z_cam", type=float, default=1.0)
    ap.add_argument("--mesh_grid", type=int, default=20)
    ap.add_argument("--min_area", type=int, default=300)
    # per-camera orientations
    ap.add_argument("--yaws", nargs="+", type=float, default=[45.0, 135.0, 225.0, 315.0])
    ap.add_argument("--pitches", nargs="+", type=float, default=[45.0, 45.0, 45.0, 45.0])
    ap.add_argument("--rolls", nargs="+", type=float, default=[0.0, 0.0, 0.0, 0.0])
    args = ap.parse_args()

    cfg = load_profile(args.config)

    # limit to 4
    videos = args.videos[:4]
    N = len(videos)

    caps = open_caps(videos)
    if all(c is None for c in caps):
        sys.exit("No valid video sources.")

    # Read first frame for each cap to get sizes; fall back to first valid cap for intrinsics
    sizes = []
    first_valid = None
    for i, cap in enumerate(caps):
        if cap is None:
            sizes.append((720, 1280))  # default
            continue
        ok, frame = cap.read()
        if not ok:
            sizes.append((720, 1280))
            continue
        H, W = frame.shape[:2]
        sizes.append((H, W))
        if first_valid is None:
            first_valid = (H, W)
        # rewind one frame if possible
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if first_valid is None:
        first_valid = sizes[0]
    H0, W0 = first_valid

    fx = args.fx if args.fx is not None else 0.8 * W0
    fy = args.fy if args.fy is not None else 0.8 * W0
    cx = args.cx if args.cx is not None else W0 / 2.0
    cy = args.cy if args.cy is not None else H0 / 2.0

    # Per-camera calibration/rotation & meshes
    cams = []
    for i in range(N):
        H, W = sizes[i]
        K = build_intrinsics(fx, fy, cx, cy)  # sharing intrinsics across cams by default
        K_inv = np.linalg.inv(K)
        yaw = args.yaws[i] if i < len(args.yaws) else args.yaws[-1]
        pitch = args.pitches[i] if i < len(args.pitches) else args.pitches[-1]
        roll = args.rolls[i] if i < len(args.rolls) else args.rolls[-1]
        R_wc = camera_rotation(yaw, pitch, roll)
        U, V, Xw, Yw, Zw = build_image_plane_mesh(W, H, K_inv, R_wc, z_cam=args.plane_z_cam, grid=args.mesh_grid)
        cams.append({
            "size": (H, W),
            "K_inv": K_inv,
            "R_wc": R_wc,
            "mesh": (Xw, Yw, Zw),
        })

    # --- Matplotlib Scene ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("Multi-Cam: Image planes + Rays")

    # Ground plane
    s = 4.5
    gx = np.linspace(-s, s, 2)
    gy = np.linspace(-s, s, 2)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.full_like(GX, -1.5)
    ground = ax.plot_surface(GX, GY, GZ, alpha=0.12, edgecolor='k')

    # Per-camera surfaces and rays
    grid = args.mesh_grid
    surf_list = []
    ray_lines = []
    # precompute sampling indices for the first camera size (shared)
    uu = np.linspace(0, W0-1, grid).astype(np.int32)
    vv = np.linspace(0, H0-1, grid).astype(np.int32)

    for i in range(N):
        Xw, Yw, Zw = cams[i]["mesh"]
        surf = ax.plot_surface(Xw, Yw, Zw, rstride=1, cstride=1, facecolors=np.ones((grid, grid, 3)), shade=False)
        surf_list.append(surf)
        (ray,) = ax.plot([0,0], [0,0], [0,0], linewidth=2)
        ray_lines.append(ray)

        # Frustum lines (corners + center)
        H, W = cams[i]["size"]
        K_inv = cams[i]["K_inv"]
        R_wc = cams[i]["R_wc"]
        corners = [(0,0), (W-1,0), (W-1,H-1), (0,H-1), (cx,cy)]
        for (u,v) in corners:
            d_cam = backproject_dir(u, v, K_inv)
            d_world = R_wc @ d_cam
            P = d_world * args.plane_z_cam
            ax.plot([0, P[0]], [0, P[1]], [0, P[2]], linewidth=1)

    # Camera axes
    axis_len = 0.6
    axes_cam = np.eye(3) * axis_len
    # Use cam0 orientation for axes display
    axes_world = cams[0]["R_wc"] @ axes_cam
    ax.scatter([0], [0], [0], s=30)
    ax.plot([0, axes_world[0,0]],[0, axes_world[1,0]],[0, axes_world[2,0]], linewidth=2)
    ax.plot([0, axes_world[0,1]],[0, axes_world[1,1]],[0, axes_world[2,1]], linewidth=2)
    ax.plot([0, axes_world[0,2]],[0, axes_world[1,2]],[0, axes_world[2,2]], linewidth=2)

    ax.set_title("Multi-Cam 3D: Textured image planes + Rays to HSV centroids")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1,1,0.6])
    ax.view_init(elev=25, azim=45)
    plt.show(block=False)

    paused = False
    while True:
        any_ok = False
        for i, cap in enumerate(caps):
            if cap is None:
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            any_ok = True

            # Compute mask & centroid
            mask = make_mask(frame, cfg)
            c = largest_contour_centroid(mask, min_area=args.min_area)

            # Update surface texture (downsample to (grid,grid))
            H, W = frame.shape[:2]
            # (recompute indices if size differs from first_valid)
            if (H, W) != (H0, W0):
                uu = np.linspace(0, W-1, grid).astype(np.int32)
                vv = np.linspace(0, H-1, grid).astype(np.int32)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tex = frame_rgb[np.ix_(vv, uu)] / 255.0
            surf_list[i].set_facecolors(tex.reshape((-1, 3)))

            # Update ray
            if c is not None:
                u, v = float(c[0]), float(c[1])
                d_cam = backproject_dir(u, v, cams[i]["K_inv"])
                d_world = cams[i]["R_wc"] @ d_cam
                L = 6.0
                P_far = d_world * L
                ray_lines[i].set_data_3d([0, P_far[0]], [0, P_far[1]], [0, P_far[2]])
            else:
                ray_lines[i].set_data_3d([0,0], [0,0], [0,0])

            # Per-cam debug windows
            vis = frame.copy()
            if c is not None:
                cv2.circle(vis, (int(c[0]), int(c[1])), 6, (0,255,0), -1)
            cv2.imshow(f"frame_{i+1}", vis)
            cv2.imshow(f"mask_{i+1}", mask)

        # Draw/update
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # space
            paused = not paused
            # simple pause: just wait in-place
            while paused:
                if cv2.waitKey(30) & 0xFF == 32:
                    paused = False
                elif cv2.waitKey(30) & 0xFF == ord('q'):
                    paused = False
                    break

        if not any_ok:
            break

    for cap in caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)


if __name__ == "__main__":
    main()
