#!/usr/bin/env python3
import argparse
import math
import csv
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

"""
3D visualization of 2D tracking points using a simple pinhole camera model.

- Camera at origin (0,0,0).
- World axes: X right, Y forward, Z up.
- Camera orientation: yaw (around Z) = +45°, pitch (around X) = +45°, roll = 0°
  (i.e., optical axis has azimuth 45° in XY and is elevated 45° above the XY plane).
- Ground plane: z = -H (default H=1.5m).

Given pixel points (u, v), we back-project a ray via K^{-1} and rotate it into world coordinates, then intersect with the ground plane.
"""


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def build_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=float)
    return K


def backproject_to_ray(u: float, v: float, K_inv: np.ndarray) -> np.ndarray:
    uv1 = np.array([u, v, 1.0], dtype=float)
    d_cam = K_inv @ uv1
    # normalize ray direction
    d_cam = d_cam / np.linalg.norm(d_cam)
    return d_cam


def intersect_with_plane(origin: np.ndarray, direction: np.ndarray, plane_z: float) -> np.ndarray:
    # Plane is z = plane_z. Solve origin + t*direction has z = plane_z -> t = (plane_z - origin_z)/dz
    dz = direction[2]
    if abs(dz) < 1e-8:
        return None
    t = (plane_z - origin[2]) / dz
    if t <= 0:
        # Intersection is behind the camera or at the origin; ignore
        return None
    return origin + t * direction


def camera_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    # R = Rz(yaw) * Rx(pitch) * Rz(roll)   (YXZ/XYZ conventions vary; this one is explicit)
    # We'll use R_world_from_cam = Rz(yaw) @ Rx(pitch) @ Rz(roll)
    R = rot_z(yaw) @ rot_x(pitch) @ rot_z(roll)
    return R


def load_points(csv_path: str) -> List[Tuple[float, float]]:
    pts = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "u" not in reader.fieldnames or "v" not in reader.fieldnames:
            raise ValueError("CSV must have headers 'u' and 'v'")
        for row in reader:
            u = float(row["u"])
            v = float(row["v"])
            pts.append((u, v))
    return pts


def draw_camera(ax, scale=0.25):
    # Simple camera icon: pyramid/frustum lines
    origin = np.zeros(3)
    # Camera look direction is R * [0,0,1] (since image plane z=1 in cam space)
    # We'll render the local camera axes for reference
    X = np.array([scale, 0, 0])
    Y = np.array([0, scale, 0])
    Z = np.array([0, 0, scale])
    # local axis lines
    ax.plot([0, X[0]], [0, X[1]], [0, X[2]], linewidth=2)
    ax.plot([0, Y[0]], [0, Y[1]], [0, Y[2]], linewidth=2)
    ax.plot([0, Z[0]], [0, Z[1]], [0, Z[2]], linewidth=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV with columns: u,v")
    ap.add_argument("--imgw", type=float, default=1280.0, help="Image width in pixels")
    ap.add_argument("--imgh", type=float, default=720.0, help="Image height in pixels")
    ap.add_argument("--fx", type=float, default=800.0, help="Focal length fx (pixels)")
    ap.add_argument("--fy", type=float, default=800.0, help="Focal length fy (pixels)")
    ap.add_argument("--cx", type=float, default=640.0, help="Principal point cx (pixels)")
    ap.add_argument("--cy", type=float, default=360.0, help="Principal point cy (pixels)")
    ap.add_argument("--yaw", type=float, default=45.0, help="Yaw around Z (degrees)")
    ap.add_argument("--pitch", type=float, default=45.0, help="Pitch around X (degrees)")
    ap.add_argument("--roll", type=float, default=0.0, help="Roll around Z (degrees)")
    ap.add_argument("--ground_z", type=float, default=-1.5, help="Ground plane z value")
    args = ap.parse_args()

    # Build camera model
    K = build_intrinsics(args.fx, args.fy, args.cx, args.cy)
    K_inv = np.linalg.inv(K)
    R = camera_rotation(args.yaw, args.pitch, args.roll)  # world_from_cam
    cam_origin = np.zeros(3)

    # Load points and unproject
    pts_px = load_points(args.csv)
    pts_world = []
    for (u, v) in pts_px:
        d_cam = backproject_to_ray(u, v, K_inv)  # in camera coords
        d_world = (R @ d_cam)  # rotate into world
        hit = intersect_with_plane(cam_origin, d_world, args.ground_z)
        if hit is not None:
            pts_world.append(hit)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Reconstruction from 2D Tracking (Ground plane intersect)")

    # Draw ground plane as a grid
    s = 3.0
    gx = np.linspace(-s, s, 2)
    gy = np.linspace(-s, s, 2)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.full_like(GX, args.ground_z)
    ax.plot_surface(GX, GY, GZ, alpha=0.2, edgecolor='k')

    # Draw camera frustum rays (for corners of the image)
    corners = [(0, 0), (args.imgw, 0), (args.imgw, args.imgh), (0, args.imgh), (args.cx, args.cy)]
    for (u, v) in corners:
        d_cam = backproject_to_ray(u, v, K_inv)
        d_world = (R @ d_cam)
        # draw a short ray
        t = 1.0
        P = cam_origin + d_world * t
        ax.plot([cam_origin[0], P[0]], [cam_origin[1], P[1]], [cam_origin[2], P[2]], linewidth=1)

    # Draw camera axes (local unit axes rotated into world)
    ax.scatter([0], [0], [0], marker='o', s=40)
    # local axes in camera space
    axis_len = 0.5
    axes_cam = np.eye(3) * axis_len
    axes_world = R @ axes_cam
    # X axis
    ax.plot([0, axes_world[0,0]], [0, axes_world[1,0]], [0, axes_world[2,0]], linewidth=2)
    # Y axis
    ax.plot([0, axes_world[0,1]], [0, axes_world[1,1]], [0, axes_world[2,1]], linewidth=2)
    # Z axis (optical axis when roll=0 and fx,fy positive)
    ax.plot([0, axes_world[0,2]], [0, axes_world[1,2]], [0, axes_world[2,2]], linewidth=2)

    # Plot 3D track
    if len(pts_world) > 0:
        X = [p[0] for p in pts_world]
        Y = [p[1] for p in pts_world]
        Z = [p[2] for p in pts_world]
        ax.plot(X, Y, Z, linewidth=2)
        ax.scatter(X, Y, Z, s=8)

    # Aesthetics
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1,1,0.5])
    ax.view_init(elev=25, azim=45)

    plt.show()


if __name__ == "__main__":
    main()
