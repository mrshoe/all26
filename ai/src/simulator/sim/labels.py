"""Labels worked out from geometry, next to the ones read from the mask:

- **Keypoints**: named points on objects (a box's corners, the tip of a
  pipe), where each lands in a camera's picture, and whether it's seen
  there. The depth image tells: a point is hidden if what's drawn at its
  pixel, and at every pixel next to it, is in front of it (by more than a
  few centimeters). Then whatever the mask shows at its pixel hides it: the
  object itself (its far side), or something in front. Otherwise it's
  seen, including a corner exactly on an outline, whose pixel may show
  what's behind it.
- **Full boxes**: an object's box as if nothing hid it, from its whole
  shape (a Prop's collision hulls), next to the box of what's seen. And
  how many pixels it would cover, so visible pixels / full pixels is the
  share of it that's seen.

Every point an object has is recorded, seen or not, so whoever trains on
them chooses which to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from sim import assets
from sim.objects import Box, Floor, Item, Prop, Solid, Sphere, Wall

# Something drawn hides a point only if it's this much nearer: a few
# centimeters, plus a little more further away, where a pixel covers more.
SEEN_WITHIN_M = 0.03
SEEN_WITHIN_SHARE = 0.01
# Points nearer the camera than this can't be drawn; parts of a shape that
# reach behind it are cut here.
NEAR = 0.01
# A sphere's outline, as points: enough that the box is within a small
# fraction of a pixel of the true one.
SPHERE_STEPS = 48


@dataclass
class Point:
    label: str  # the keypoint's class, e.g. "corner"
    index: int  # which of the object's points of that class
    x: float
    y: float
    depth: float  # meters in front of the camera, along its axis
    in_frame: bool
    visible: bool
    # The id of what hides it (the object's own id when its far side does),
    # or None if it's seen or out of the picture.
    occluded_by: Optional[int]


class View:
    """Where a camera is, and its pinhole intrinsics (geometry.intrinsics)."""

    def __init__(self, position, quat, k: dict):
        self.position = np.asarray(position, float)
        self.turn = Rotation.from_quat(quat, scalar_first=True)
        self.k = k
        self.width, self.height = int(k["width"]), int(k["height"])

    def local(self, points: np.ndarray) -> np.ndarray:
        """World points in the camera's frame: x right, y up, looking along -z."""
        return self.turn.inv().apply(np.asarray(points, float).reshape(-1, 3) - self.position)

    def pixels(self, local: np.ndarray) -> np.ndarray:
        """Camera-frame points (in front of it) as pixels (x, y)."""
        depth = -local[:, 2]
        k = self.k
        return np.column_stack([k["cx"] + k["fx"] * local[:, 0] / depth, k["cy"] - k["fy"] * local[:, 1] / depth])


# --- Where things are ---------------------------------------------------------


def pose(item: Item) -> tuple[np.ndarray, Rotation]:
    """Where an object's own coordinates are in the world now: the frame
    its keypoints and shape are given in."""
    if isinstance(item, Wall):
        center, _, yaw = item.box()
        return center, Rotation.from_euler("z", yaw, degrees=True)
    if isinstance(item, Floor):
        return np.zeros(3), Rotation.identity()
    return np.asarray(item.position, float), Rotation.from_quat(item.quat, scalar_first=True)


def to_world(item: Item, points) -> np.ndarray:
    position, turn = pose(item)
    return turn.apply(np.asarray(points, float).reshape(-1, 3)) + position


def keypoints_of(item: Item) -> dict[str, np.ndarray]:
    """The object's keypoints by class, in its own coordinates: a Prop's from
    the catalog (scaled), then its own `keypoints`, which replace a class of
    the same name."""
    out: dict[str, np.ndarray] = {}
    if isinstance(item, Prop):
        for name, points in assets.info(item.asset).get("keypoints", {}).items():
            out[name] = np.asarray(points, float).reshape(-1, 3) * item.scale
    for name, points in (getattr(item, "keypoints", None) or {}).items():
        out[name] = np.asarray(points, float).reshape(-1, 3)
    return out


def box_corners(size) -> np.ndarray:
    """A box's 8 corners around its center."""
    half = np.asarray(size, float) / 2
    return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half


def sphere_points(radius: float) -> np.ndarray:
    """Points around a sphere, pushed out a little so the polygon they make
    holds the whole sphere (a circumscribed one, not an inscribed one)."""
    n = SPHERE_STEPS
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    b = np.linspace(-np.pi / 2, np.pi / 2, n // 2 + 1)
    lon, lat = np.meshgrid(a, b)
    unit = np.column_stack([(np.cos(lat) * np.cos(lon)).ravel(), (np.cos(lat) * np.sin(lon)).ravel(),
                            np.sin(lat).ravel()])
    return unit * radius / np.cos(np.pi / n)


def shape(item: Solid) -> list[np.ndarray]:
    """The object's whole shape as convex parts, each an (n, 3) array of
    points in its own coordinates."""
    if isinstance(item, Sphere):
        return [sphere_points(item.radius)]
    if isinstance(item, Wall):
        return [box_corners(item.box()[1])]
    if isinstance(item, Floor):
        return [box_corners((*item.size[:2], 0))]
    if isinstance(item, Prop):
        try:
            return [np.asarray(p, float) * item.scale for p in assets.collision(item.asset)]
        except assets.MissingAsset:
            if assets.cloud_bucket():
                raise
            # On a laptop without its files, its box from the catalog, as physics does.
            size = item.size()
            return [box_corners(size) + [0, 0, size[2] / 2]]
    if isinstance(item, Box):
        return [box_corners(item.size)]
    raise TypeError(f"There's no shape for a {type(item).__name__}")


# --- In a picture -------------------------------------------------------------


def outline(view: View, part: np.ndarray) -> Optional[np.ndarray]:
    """A convex part's outline in the picture: the pixels of the points of
    its convex hull. Where it reaches behind the camera, it's cut at NEAR.
    None if it's entirely behind."""
    local = view.local(part)
    depth = -local[:, 2]
    front, back = local[depth >= NEAR], local[depth < NEAR]
    if not len(front):
        return None
    if len(back):
        # The hull of the part cut at NEAR: its points in front, and where
        # each line from one in front to one behind crosses NEAR.
        f, b = front[:, None, :], back[None, :, :]
        df, db = -f[..., 2], -b[..., 2]
        t = (df - NEAR) / (df - db)
        front = np.vstack([front, (f + t[..., None] * (b - f)).reshape(-1, 3)])
    return view.pixels(front)


def full_box(view: View, item: Solid, parts: Optional[list[np.ndarray]] = None) -> Optional[tuple]:
    """The box (x0, y0, x1, y1) of an object's whole shape where it is now,
    cut to the picture, and how many pixels it would cover if nothing hid
    it. None if none of it is in the picture. `parts` is shape(item), if
    it's already worked out."""
    parts = shape(item) if parts is None else parts
    polygons = [p for p in (outline(view, to_world(item, part)) for part in parts) if p is not None]
    if not polygons:
        return None
    # A part's outline is the convex hull of its points' pixels.
    hulls = [hull(p) for p in polygons]
    w, h = view.width, view.height
    points = np.vstack(hulls)
    x0, y0 = np.clip(points.min(0), 0, [w, h])
    x1, y1 = np.clip(points.max(0), 0, [w, h])
    if x1 <= x0 or y1 <= y0:
        return None
    covered = np.zeros((h, w), bool)
    for poly in hulls:
        fill(covered, poly)
    pixels = int(np.count_nonzero(covered))
    return float(x0), float(y0), float(x1), float(y1), pixels


def fill(covered: np.ndarray, poly: np.ndarray) -> None:
    """Mark the pixels whose centers are inside a convex polygon, as the
    mask counts them."""
    h, w = covered.shape
    (x0, y0), (x1, y1) = np.floor(poly.min(0)), np.ceil(poly.max(0))
    x0, x1 = int(max(x0, 0)), int(min(x1, w))
    y0, y1 = int(max(y0, 0)), int(min(y1, h))
    if len(poly) < 3 or x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1] + 0.5
    inside = np.ones(xs.shape, bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        # hull() goes counterclockwise (with y up), so inside is to the left of each edge.
        inside &= (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0]) >= 0
    covered[y0:y1, x0:x1] |= inside


def hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of 2D points, counterclockwise (Qhull's, for 2D)."""
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except QhullError:  # all on one line, or all one point: it has no inside
        return pts[[pts[:, 0].argmin(), pts[:, 0].argmax()]]


def keypoints(view: View, item: Item, defined: dict[str, np.ndarray], ids: np.ndarray,
              depth: np.ndarray) -> list[Point]:
    """Each of an object's keypoints in this picture. Points behind the
    camera have no pixel, so they're left out."""
    out = []
    for label, points in defined.items():
        local = view.local(to_world(item, points))
        for index, p in enumerate(local):
            z = -p[2]
            if z < NEAR:
                continue
            x, y = view.pixels(p[None])[0]
            in_frame = 0 <= x < view.width and 0 <= y < view.height
            visible, by = False, None
            if in_frame:
                visible, by = seen(int(x), int(y), z, ids, depth)
            out.append(Point(label, index, float(x), float(y), float(z), bool(in_frame), visible, by))
    return out


def seen(col: int, row: int, z: float, ids: np.ndarray, depth: np.ndarray) -> tuple[bool, Optional[int]]:
    """Whether a point at this pixel and depth is seen, and if not, the id
    of what's drawn in front of it."""
    near = depth[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
    if near.max() >= z - (SEEN_WITHIN_M + SEEN_WITHIN_SHARE * z):
        return True, None
    return False, int(ids[row, col])
