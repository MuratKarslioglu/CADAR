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


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def build_lookat_rotation(cam_pos, target=np.zeros(3)):
    # Camera forward (optical axis) is +Z_cam -> map to (target - pos) in world
    f = normalize(target - cam_pos)
    up_ref = np.array([0.0, 0.0, 1.0])
    # If forward nearly colinear with up, choose another up
    if abs(np.dot(f, up_ref)) > 0.999:
        up_ref = np.array([0.0, 1.0, 0.0])
    x = normalize(np.cross(up_ref, f))   # right
    y = normalize(np.cross(f, x))        # real up
    z = f                                # forward
    R_wc = np.column_stack([x, y, z])    # world_from_cam
    return R_wc


def build_image_plane_mesh(W, H, K_inv, R_wc, cam_pos, z_cam=1.0, grid=20):
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
            Pw = cam_pos + (R_wc @ P_cam)
            Xw[i, j], Yw[i, j], Zw[i, j] = Pw[0], Pw[1], Pw[2]
    return Xw, Yw, Zw


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
    ap.add_argument("--videos", nargs="+", required=True, help="Up to 4 paths. 'webcam' or '0' allowed.")
    ap.add_argument("--config", type=str, default="profile.json")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--plane_z_cam", type=float, default=1.0, help="Depth of image plane in camera coords")
    ap.add_argument("--mesh_grid", type=int, default=20)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--layout", type=str, default="inward", choices=["inward","outward"])
    ap.add_argument("--radius", type=float, default=3.0, help="Camera ring radius around origin")
    ap.add_argument("--heights", nargs="+", type=float, default=[1.0,1.0,1.0,1.0], help="Per-camera z (m)")
    ap.add_argument("--angles", nargs="+", type=float, default=[135.0,135.0,135.0,135.0],
                    help="Angular placement (deg) around the ring for each camera")
    args = ap.parse_args()

    cfg = load_profile(args.config)

    videos = args.videos[:4]
    N = len(videos)
    caps = open_caps(videos)
    if all(c is None for c in caps):
        sys.exit("No valid video sources.")

    # Read one frame per cap (to know sizes)
    sizes = []
    first_valid = None
    for i, cap in enumerate(caps):
        if cap is None:
            sizes.append((720, 1280))
            continue
        ok, frame = cap.read()
        if not ok:
            sizes.append((720, 1280))
            continue
        H, W = frame.shape[:2]
        sizes.append((H, W))
        if first_valid is None:
            first_valid = (H, W)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if first_valid is None:
        first_valid = sizes[0]
    H0, W0 = first_valid

    fx = args.fx if args.fx is not None else 0.8 * W0
    fy = args.fy if args.fy is not None else 0.8 * W0
    cx = args.cx if args.cx is not None else W0 / 2.0
    cy = args.cy if args.cy is not None else H0 / 2.0
    K = build_intrinsics(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)

    # Build per-camera pose on a ring
    cams = []
    for i in range(N):
        ang_deg = args.angles[i] if i < len(args.angles) else args.angles[-1]
        ang = math.radians(ang_deg)
        z = args.heights[i] if i < len(args.heights) else args.heights[-1]
        cam_pos = np.array([args.radius*math.cos(ang), args.radius*math.sin(ang), z], float)
        if args.layout == "inward":
            R_wc = build_lookat_rotation(cam_pos, target=np.zeros(3))
        else:
            # outward: look away from center
            R_wc = build_lookat_rotation(cam_pos, target=cam_pos + (cam_pos - np.zeros(3)))
        Xw, Yw, Zw = build_image_plane_mesh(sizes[i][1], sizes[i][0], K_inv, R_wc, cam_pos,
                                            z_cam=args.plane_z_cam, grid=args.mesh_grid)
        cams.append({
            "pos": cam_pos,
            "R_wc": R_wc,
            "K_inv": K_inv,
            "size": (sizes[i][0], sizes[i][1]),
            "mesh": (Xw, Yw, Zw),
        })

    # --- Scene ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("Multi-Cam (Inward/Outward): Textured image planes + Rays")

    # Ground plane
    s = args.radius + 2.0
    gx = np.linspace(-s, s, 2)
    gy = np.linspace(-s, s, 2)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)  # ground at z=0
    ground = ax.plot_surface(GX, GY, GZ, alpha=0.12, edgecolor='k')

    # Camera surfaces, rays, and axes
    grid = args.mesh_grid
    surf_list, ray_lines = [], []
    for i in range(N):
        Xw, Yw, Zw = cams[i]["mesh"]
        surf = ax.plot_surface(Xw, Yw, Zw, rstride=1, cstride=1,
                               facecolors=np.ones((grid, grid, 3)), shade=False)
        surf_list.append(surf)
        (ray,) = ax.plot([0,0], [0,0], [0,0], linewidth=2)
        ray_lines.append(ray)

        # Frustum reference lines (corners + center)
        H, W = cams[i]["size"]
        K_inv_i = cams[i]["K_inv"]
        R_wc_i = cams[i]["R_wc"]
        C_i = cams[i]["pos"]
        corners = [(0,0), (W-1,0), (W-1,H-1), (0,H-1), (cx,cy)]
        for (u,v) in corners:
            d_cam = backproject_dir(u, v, K_inv_i)
            d_world = R_wc_i @ d_cam
            P = C_i + d_world * args.plane_z_cam
            ax.plot([C_i[0], P[0]], [C_i[1], P[1]], [C_i[2], P[2]], linewidth=1)

        # Camera axes
        axis_len = 0.3
        axes_cam = np.eye(3) * axis_len
        axes_world = R_wc_i @ axes_cam
        ax.scatter([C_i[0]], [C_i[1]], [C_i[2]], s=25)
        ax.plot([C_i[0], C_i[0]+axes_world[0,0]],
                [C_i[1], C_i[1]+axes_world[1,0]],
                [C_i[2], C_i[2]+axes_world[2,0]], linewidth=2)
        ax.plot([C_i[0], C_i[0]+axes_world[0,1]],
                [C_i[1], C_i[1]+axes_world[1,1]],
                [C_i[2], C_i[2]+axes_world[2,1]], linewidth=2)
        ax.plot([C_i[0], C_i[0]+axes_world[0,2]],
                [C_i[1], C_i[1]+axes_world[1,2]],
                [C_i[2], C_i[2]+axes_world[2,2]], linewidth=2)

    ax.set_title("Multi-Cam 3D (Inward/Outward): Textured image planes + Rays to HSV centroids")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1,1,0.5])
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

            mask = make_mask(frame, cfg)
            c = largest_contour_centroid(mask, min_area=args.min_area)

            # Update texture (safe resize to grid x grid)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tex = cv2.resize(frame_rgb, (grid, grid), interpolation=cv2.INTER_AREA) / 255.0
            surf_list[i].set_facecolors(tex.reshape((-1, 3)))

            # Update ray for this cam
            if c is not None:
                u, v = float(c[0]), float(c[1])
                d_cam = backproject_dir(u, v, cams[i]["K_inv"])
                d_world = cams[i]["R_wc"] @ d_cam
                C_i = cams[i]["pos"]
                L = args.radius * 2.0
                P_far = C_i + d_world * L
                ray_lines[i].set_data_3d([C_i[0], P_far[0]], [C_i[1], P_far[1]], [C_i[2], P_far[2]])
            else:
                C_i = cams[i]["pos"]
                ray_lines[i].set_data_3d([C_i[0], C_i[0]], [C_i[1], C_i[1]], [C_i[2], C_i[2]])

            # 2D debug
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
            while paused:
                k2 = cv2.waitKey(30) & 0xFF
                if k2 == 32:
                    paused = False
                elif k2 == ord('q'):
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
