#!/usr/bin/env python3
import argparse
import json
import math
from typing import Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    # ensure odd blur
    if cfg["blur"] > 0 and cfg["blur"] % 2 == 0:
        cfg["blur"] += 1
    if cfg["kernel"] < 1:
        cfg["kernel"] = 1
    return cfg


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def camera_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float=0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    # world_from_cam rotation
    R = rot_z(yaw) @ rot_x(pitch) @ rot_z(roll)
    return R


def build_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1]], dtype=float)


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
        edges = cv2.Canny(mask, c1, c2)
        mask = cv2.bitwise_or(mask, edges)
    return mask


def largest_contour_centroid(mask, min_area=300) -> Optional[Tuple[int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def backproject_dir(u: float, v: float, K_inv: np.ndarray) -> np.ndarray:
    uv1 = np.array([u, v, 1.0], dtype=float)
    d_cam = K_inv @ uv1
    d_cam = d_cam / np.linalg.norm(d_cam)
    return d_cam


def intersect_z_plane(origin: np.ndarray, direction: np.ndarray, z_plane: float) -> Optional[np.ndarray]:
    dz = direction[2]
    if abs(dz) < 1e-9:
        return None
    t = (z_plane - origin[2]) / dz
    if t <= 0:
        return None
    return origin + t * direction


def build_image_plane_mesh(W: int, H: int, K_inv: np.ndarray, R_wc: np.ndarray, z_cam: float=1.0, grid: int=20):
    # Build a coarse grid of pixel coordinates and map to world points on plane z_cam (cam coords)
    uu = np.linspace(0, W-1, grid)
    vv = np.linspace(0, H-1, grid)
    U, V = np.meshgrid(uu, vv)
    Xw = np.zeros_like(U, dtype=float)
    Yw = np.zeros_like(U, dtype=float)
    Zw = np.zeros_like(U, dtype=float)

    for i in range(grid):
        for j in range(grid):
            u = U[i, j]
            v = V[i, j]
            d_cam = backproject_dir(u, v, K_inv)
            # intersection with plane z=z_cam in camera coordinates -> t = z_cam / d_cam_z (since cam at origin)
            if abs(d_cam[2]) < 1e-9:
                P_cam = np.array([0,0,z_cam], dtype=float)  # fallback
            else:
                t = z_cam / d_cam[2]
                P_cam = d_cam * t
            # to world
            P_world = R_wc @ P_cam
            Xw[i, j], Yw[i, j], Zw[i, j] = P_world[0], P_world[1], P_world[2]

    return Xw, Yw, Zw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="k1.mp4", help="Path to video. If omitted, uses webcam 0.")
    ap.add_argument("--config", type=str, default="profile.json", help="HSV profile JSON path.")
    ap.add_argument("--fx", type=float, default=None, help="fx (pixels). Default: 0.8*W")
    ap.add_argument("--fy", type=float, default=None, help="fy (pixels). Default: 0.8*W")
    ap.add_argument("--cx", type=float, default=None, help="cx (pixels). Default: W/2")
    ap.add_argument("--cy", type=float, default=None, help="cy (pixels). Default: H/2")
    ap.add_argument("--yaw", type=float, default=45.0, help="Yaw around Z in degrees")
    ap.add_argument("--pitch", type=float, default=45.0, help="Pitch around X in degrees")
    ap.add_argument("--roll", type=float, default=0.0, help="Roll around Z in degrees")
    ap.add_argument("--plane_z_cam", type=float, default=1.0, help="Depth of the image plane in camera coords (meters)")
    ap.add_argument("--min_area", type=int, default=300, help="Min contour area to accept centroid")
    ap.add_argument("--mesh_grid", type=int, default=20, help="Grid resolution of the textured image plane")
    args = ap.parse_args()

    cfg = load_profile(args.config)

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame.")
    H, W = frame.shape[:2]

    fx = args.fx if args.fx is not None else 0.8 * W
    fy = args.fy if args.fy is not None else 0.8 * W
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0

    K = build_intrinsics(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)

    R_wc = camera_rotation(args.yaw, args.pitch, args.roll)  # world_from_cam
    cam_origin = np.zeros(3)

    # Precompute image plane mesh topology (XY, but colors update every frame)
    Xw, Yw, Zw = build_image_plane_mesh(W, H, K_inv, R_wc, z_cam=args.plane_z_cam, grid=args.mesh_grid)

    # Matplotlib init
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D: Camera (0,0,0) @ yaw=45°, pitch=45° | Image plane + Ray to tracked point")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1,1,0.6])

    paused = False
    ray_line = None
    surf = None
    frustum_lines = []

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            mask = make_mask(frame, cfg)
            c = largest_contour_centroid(mask, min_area=args.min_area)

            # Clear axes
            ax.cla()
            ax.set_title("3D: Camera (0,0,0) @ yaw=45°, pitch=45° | Image plane + Ray to tracked point")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_box_aspect([1,1,0.6])

            # Draw ground-ish grid (optional): simple square at z = -1.5
            s = 3.0
            gx = np.linspace(-s, s, 2)
            gy = np.linspace(-s, s, 2)
            GX, GY = np.meshgrid(gx, gy)
            GZ = np.full_like(GX, -1.5)
            ax.plot_surface(GX, GY, GZ, alpha=0.15, edgecolor='k')

            # Draw camera axes
            axis_len = 0.5
            axes_cam = np.eye(3) * axis_len
            axes_world = R_wc @ axes_cam
            ax.scatter([0], [0], [0], s=20)
            ax.plot([0, axes_world[0,0]], [0, axes_world[1,0]], [0, axes_world[2,0]], linewidth=2)
            ax.plot([0, axes_world[0,1]], [0, axes_world[1,1]], [0, axes_world[2,1]], linewidth=2)
            ax.plot([0, axes_world[0,2]], [0, axes_world[1,2]], [0, axes_world[2,2]], linewidth=2)

            # Update image plane colors from current frame (downsample to mesh grid)
            # Build facecolors by sampling the frame at grid coordinates
            grid = args.mesh_grid
            uu = np.linspace(0, W-1, grid).astype(np.int32)
            vv = np.linspace(0, H-1, grid).astype(np.int32)
            # Create an array of colors normalized to [0,1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tex = frame_rgb[np.ix_(vv, uu)] / 255.0  # (grid, grid, 3)

            # plot the textured image plane
            surf = ax.plot_surface(Xw, Yw, Zw, rstride=1, cstride=1, facecolors=tex, shade=False)

            # Draw frustum rays from the 4 corners + center (for reference)
            corners = [(0,0), (W-1,0), (W-1,H-1), (0,H-1), (cx,cy)]
            for (u,v) in corners:
                d_cam = backproject_dir(u, v, K_inv)
                d_world = R_wc @ d_cam
                P = cam_origin + d_world * args.plane_z_cam  # short line to image plane depth
                ax.plot([cam_origin[0], P[0]], [cam_origin[1], P[1]], [cam_origin[2], P[2]], linewidth=1)

            # If we have a tracked centroid, cast a ray
            if c is not None:
                u, v = float(c[0]), float(c[1])
                d_cam = backproject_dir(u, v, K_inv)
                d_world = R_wc @ d_cam
                # Draw a ray segment forward (length L) and also to ground plane intersection
                L = 5.0
                P_far = cam_origin + d_world * L
                ax.plot([cam_origin[0], P_far[0]], [cam_origin[1], P_far[1]], [cam_origin[2], P_far[2]], linewidth=2)

                # Mark the intersection with the image plane depth (for reference)
                if abs(d_cam[2]) > 1e-9:
                    t_img = args.plane_z_cam / d_cam[2]
                    P_img = cam_origin + (R_wc @ d_cam) * t_img
                    ax.scatter([P_img[0]], [P_img[1]], [P_img[2]], s=30)

            # Set a consistent view
            ax.view_init(elev=25, azim=45)

            # Also show the current 2D frame & mask side-by-side
            vis = frame.copy()
            if c is not None:
                cv2.circle(vis, (int(c[0]), int(c[1])), 6, (0,255,0), -1)
            cv2.imshow("frame", vis)
            cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # space to pause/resume
            paused = not paused
        plt.pause(0.001)

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
