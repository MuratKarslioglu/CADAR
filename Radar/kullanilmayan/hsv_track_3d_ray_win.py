#!/usr/bin/env python3
import argparse
import json
import math
import sys

# --- Choose a safe interactive backend (Windows-friendly) ---
import matplotlib
for candidate in ("QtAgg", "Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(candidate, force=True)
        break
    except Exception:
        continue
# If none works, Agg will be used by default (non-interactive).

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
    d_cam /= np.linalg.norm(d_cam)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="k1.mp4")
    ap.add_argument("--config", type=str, default="profile.json")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=45.0)
    ap.add_argument("--pitch", type=float, default=45.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--plane_z_cam", type=float, default=1.0)
    ap.add_argument("--mesh_grid", type=int, default=20)
    ap.add_argument("--min_area", type=int, default=300)
    args = ap.parse_args()

    cfg = load_profile(args.config)

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        sys.exit("Could not open video source.")

    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read first frame.")
    H, W = frame.shape[:2]

    fx = args.fx if args.fx is not None else 0.8 * W
    fy = args.fy if args.fy is not None else 0.8 * W
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    K = build_intrinsics(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)

    R_wc = camera_rotation(args.yaw, args.pitch, args.roll)
    cam_origin = np.zeros(3)

    # Precompute image plane mesh coordinates (static geometry)
    U, V, Xw, Yw, Zw = build_image_plane_mesh(W, H, K_inv, R_wc, z_cam=args.plane_z_cam, grid=args.mesh_grid)

    # --- Matplotlib figure (non-blocking) ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("3D Ray + Textured Image Plane")

    # Ground plane
    s = 3.0
    gx = np.linspace(-s, s, 2)
    gy = np.linspace(-s, s, 2)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.full_like(GX, -1.5)
    ground = ax.plot_surface(GX, GY, GZ, alpha=0.15, edgecolor='k')

    # Image plane surface (we'll only update its facecolors each frame)
    surf = ax.plot_surface(Xw, Yw, Zw, rstride=1, cstride=1, facecolors=np.ones((args.mesh_grid, args.mesh_grid, 3)), shade=False)

    # Frustum reference lines (corners + center)
    corners = [(0,0), (W-1,0), (W-1,H-1), (0,H-1), (cx,cy)]
    frustum_lines = []
    for (u,v) in corners:
        d_cam = backproject_dir(u, v, K_inv)
        d_world = R_wc @ d_cam
        P = cam_origin + d_world * args.plane_z_cam
        (line,) = ax.plot([cam_origin[0], P[0]], [cam_origin[1], P[1]], [cam_origin[2], P[2]], linewidth=1)
        frustum_lines.append(line)

    # Camera axes
    axis_len = 0.5
    axes_cam = np.eye(3) * axis_len
    axes_world = R_wc @ axes_cam
    ax.scatter([0], [0], [0], s=20)
    ax.plot([0, axes_world[0,0]], [0, axes_world[1,0]], [0, axes_world[2,0]], linewidth=2)
    ax.plot([0, axes_world[0,1]], [0, axes_world[1,1]], [0, axes_world[2,1]], linewidth=2)
    ax.plot([0, axes_world[0,2]], [0, axes_world[1,2]], [0, axes_world[2,2]], linewidth=2)

    # Ray line artist
    (ray_line,) = ax.plot([0,0], [0,0], [0,0], linewidth=2)

    ax.set_title("3D: Image plane + Ray to HSV-tracked point")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1,1,0.6])
    ax.view_init(elev=25, azim=45)
    plt.show(block=False)

    # Precompute sample indices for texture downsampling
    grid = args.mesh_grid
    uu = np.linspace(0, W-1, grid).astype(np.int32)
    vv = np.linspace(0, H-1, grid).astype(np.int32)

    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            mask = make_mask(frame, cfg)
            c = largest_contour_centroid(mask, min_area=args.min_area)

            # Update texture colors
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tex = frame_rgb[np.ix_(vv, uu)] / 255.0  # (grid, grid, 3)
            surf.set_facecolors(tex.reshape((-1, 3)))

            # Update ray if we have a centroid
            if c is not None:
                u, v = float(c[0]), float(c[1])
                d_cam = backproject_dir(u, v, K_inv)
                d_world = R_wc @ d_cam
                L = 5.0
                P_far = cam_origin + d_world * L
                ray_line.set_data_3d([cam_origin[0], P_far[0]], [cam_origin[1], P_far[1]], [cam_origin[2], P_far[2]])
            else:
                ray_line.set_data_3d([0,0], [0,0], [0,0])

            # Draw/update the figure
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # Show 2D debug windows
            vis = frame.copy()
            if c is not None:
                cv2.circle(vis, (int(c[0]), int(c[1])), 6, (0,255,0), -1)
            cv2.imshow("frame", vis)
            cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)


if __name__ == "__main__":
    main()
