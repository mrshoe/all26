"""Rotations (with scipy) and camera projection.

Quaternions are (w, x, y, z), as MuJoCo, Blender and Isaac Sim use. Euler
angles are in degrees, applied about x, then y, then z (Blender's default
"XYZ" order; scipy's "xyz"). Cameras look along their own -z axis with +y
up, as in Blender, USD and Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def quat_from_euler(degrees) -> np.ndarray:
    return Rotation.from_euler("xyz", degrees, degrees=True).as_quat(canonical=True, scalar_first=True)


def euler_from_quat(q) -> tuple[float, float, float]:
    return tuple(float(a) for a in Rotation.from_quat(q, scalar_first=True).as_euler("xyz", degrees=True))


def spin_between(q0, q1, dt: float) -> np.ndarray:
    """The angular velocity (rad/s, world frame) that turns q0 into q1 in dt."""
    r0 = Rotation.from_quat(q0, scalar_first=True)
    r1 = Rotation.from_quat(q1, scalar_first=True)
    return (r1 * r0.inv()).as_rotvec() / dt


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
    matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(matrix).as_quat(canonical=True, scalar_first=True)


def intrinsics(resolution, fov) -> dict:
    """Pinhole intrinsics for a horizontal field of view in degrees."""
    width, height = resolution
    fx = (width / 2) / math.tan(math.radians(fov) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2, "width": width, "height": height}


def project(point, camera_position, camera_quat, k: dict):
    """Pixel (x, y) of a world point, or None if it's behind the camera.
    (0, 0) is the top-left corner of the image."""
    camera = Rotation.from_quat(camera_quat, scalar_first=True)
    rel = camera.inv().apply(np.asarray(point, dtype=float) - np.asarray(camera_position, dtype=float))
    depth = -rel[2]
    if depth <= 1e-6:
        return None
    return (k["cx"] + k["fx"] * rel[0] / depth, k["cy"] - k["fy"] * rel[1] / depth)
