"""Rotations and camera projection, with numpy only.

Quaternions are (w, x, y, z). Euler angles are in degrees, applied about x,
then y, then z (Blender's default "XYZ" order). Cameras look along their own
-z axis with +y up, as in Blender, USD and Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np


def quat_from_euler(degrees) -> np.ndarray:
    rx, ry, rz = (math.radians(a) / 2 for a in degrees)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    # q = qz * qy * qx
    return np.array([
        cz * cy * cx + sz * sy * sx,
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
    ])


def euler_from_quat(q) -> tuple[float, float, float]:
    m = matrix_from_quat(q)
    ry = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 0.999999:
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: put it all in x
        rx = math.atan2(-m[1, 2], m[1, 1])
        rz = 0.0
    return tuple(math.degrees(a) for a in (rx, ry, rz))


def matrix_from_quat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def quat_looking_at(eye, target) -> np.ndarray:
    """A camera at `eye` looking at `target`, with +z up in the world."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("A camera can't look at its own position")
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.9999:  # straight up or down: pick another up
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # Columns are the camera's x (right), y (up) and z (backwards) axes.
    return quat_from_matrix(np.column_stack([right, true_up, -forward]))


def intrinsics(resolution, fov) -> dict:
    """Pinhole intrinsics for a horizontal field of view in degrees."""
    width, height = resolution
    fx = (width / 2) / math.tan(math.radians(fov) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2, "width": width, "height": height}


def project(point, camera_position, camera_quat, k: dict):
    """Pixel (x, y) of a world point, or None if it's behind the camera.
    (0, 0) is the top-left corner of the image."""
    rel = matrix_from_quat(camera_quat).T @ (np.asarray(point, dtype=float) - np.asarray(camera_position, dtype=float))
    depth = -rel[2]
    if depth <= 1e-6:
        return None
    return (k["cx"] + k["fx"] * rel[0] / depth, k["cy"] - k["fy"] * rel[1] / depth)
