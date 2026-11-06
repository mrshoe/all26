"""`Logger`: writes an episode's data to a folder. The simulator and the
robot both use it, and `episodes.Episode` reads it back.

    episode.json             what ran (whatever the caller says), the frame
                             count, and which tables there are
    <table>.parquet          the tables below, one file each
    cam/<camera>/000000.jpg  what each camera saw
    mask/<camera>/000000.png which object each pixel shows, 16-bit object
                             ids (simulation only; real cameras have none)

Every table has fixed columns (SCHEMAS). Tables join on `frame` and
`object`; an object's id is the one its mask pixels hold, and a camera is
an object too.

    objects  one row per object: object, name, type, label, static, settings
    cameras  one row per camera: object, width, height, fx, fy, cx, cy, fov
             (the intrinsics may be empty, e.g. for photos)
    sim      one row per frame: frame, time, and how long each part took
    poses    one row per object per frame (static objects: frame 0 only):
             position, rotation, linear and angular velocity
    images   one row per picture: frame, camera, time, image and mask paths
    boxes    one row per box seen: frame, camera, object (null for a model's
             detection), label, x0 y0 x1 y1 in pixels ((x1, y1) is one past
             the bottom-right corner), pixels, score (null for ground
             truth), source ("truth" or the model's name). In simulation,
             also the full box: full_x0 full_y0 full_x1 full_y1, the box of
             the object's whole shape as if nothing hid it (cut to the
             picture, like the visible one), and full_pixels, how many
             pixels it would cover; pixels / full_pixels is the share seen
    keypoints
             one row per named point on an object, per camera, per frame,
             seen or hidden (but not behind the camera): frame, camera,
             object (null for a model's), label (the point's class, e.g.
             "corner"), index (which of the object's points of that class),
             x y in pixels, depth in meters along the camera's axis,
             in_frame, visible, occluded_by (null if seen; the object's own
             id when its own far side hides it; otherwise the id of what's
             in front), score (null for ground truth), source

A robot logs its own per-frame table (frame, time, its own columns) in place
of `sim`: pass its schema to Logger.

Streaming to the visualizer will be a second place rows go, next to disk.

Needs numpy, pillow and pyarrow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

FORMAT = 2
JPEG_QUALITY = 90

i32, f32, f64, text, flag = pa.int32(), pa.float32(), pa.float64(), pa.string(), pa.bool_()


def schema(*columns, key: tuple[str, ...]) -> pa.Schema:
    """Columns are (name, type) or (name, type, "null") for ones that may be
    empty. `key` is the sort order."""
    fields = [pa.field(c[0], c[1], nullable=len(c) > 2) for c in columns]
    return pa.schema(fields, metadata={"key": ",".join(key)})


SCHEMAS: dict[str, pa.Schema] = {
    "objects": schema(
        ("object", i32), ("name", text), ("type", text), ("label", text, "null"),
        ("static", flag), ("settings", text),
        key=("object",)),
    "cameras": schema(
        ("object", i32), ("width", i32), ("height", i32),
        ("fx", f64, "null"), ("fy", f64, "null"), ("cx", f64, "null"), ("cy", f64, "null"), ("fov", f64, "null"),
        key=("object",)),
    "sim": schema(
        ("frame", i32), ("time", f64),
        ("step_s", f64), ("physics_s", f64), ("render_s", f64), ("log_s", f64),
        ("model_ms", f64, "null"),
        key=("frame",)),
    "poses": schema(
        ("frame", i32), ("object", i32),
        ("x", f64), ("y", f64), ("z", f64),
        ("qw", f64), ("qx", f64), ("qy", f64), ("qz", f64),
        ("vx", f64), ("vy", f64), ("vz", f64),
        ("wx", f64), ("wy", f64), ("wz", f64),
        key=("frame", "object")),
    "images": schema(
        ("frame", i32), ("camera", i32), ("time", f64), ("image", text), ("mask", text, "null"),
        key=("frame", "camera")),
    "boxes": schema(
        ("frame", i32), ("camera", i32), ("object", i32, "null"), ("label", text),
        ("x0", f32), ("y0", f32), ("x1", f32), ("y1", f32),
        ("pixels", i32, "null"), ("score", f32, "null"), ("source", text),
        ("full_x0", f32, "null"), ("full_y0", f32, "null"), ("full_x1", f32, "null"), ("full_y1", f32, "null"),
        ("full_pixels", i32, "null"),
        key=("frame", "camera", "object")),
    "keypoints": schema(
        ("frame", i32), ("camera", i32), ("object", i32, "null"), ("label", text), ("index", i32),
        ("x", f32), ("y", f32), ("depth", f32, "null"), ("in_frame", flag), ("visible", flag, "null"),
        ("occluded_by", i32, "null"), ("score", f32, "null"), ("source", text),
        key=("frame", "camera", "object", "label", "index")),
}


class Logger:
    def __init__(self, folder, info: Optional[dict] = None,
                 schemas: Optional[Mapping[str, pa.Schema]] = None):
        """`info` goes in episode.json (`finish` can add more). `schemas`
        adds tables, or replaces these."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.info = dict(info or {})
        self.schemas = {**SCHEMAS, **(schemas or {})}
        self.rows: dict[str, list[dict]] = {}
        self.frames: set[int] = set()

    def add(self, table: str, *rows: dict) -> None:
        """Add rows to a table. Each row has the table's columns; ones that
        may be empty can be left out."""
        if table not in self.schemas:
            raise KeyError(f"There's no table {table!r}; the tables are {', '.join(self.schemas)}")
        columns = self.schemas[table]
        for row in rows:
            extra = set(row) - set(columns.names)
            if extra:
                raise KeyError(f"{table} has no column {', '.join(sorted(extra))}")
            missing = [f.name for f in columns if not f.nullable and row.get(f.name) is None]
            if missing:
                raise ValueError(f"{table} needs {', '.join(missing)}")
            if "frame" in row:
                self.frames.add(int(row["frame"]))
            self.rows.setdefault(table, []).append(row)

    def image(self, frame: int, camera: int, name: str, time: float,
              rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Save a camera's picture (height x width x 3 uint8) and, in
        simulation, its object-id mask (height x width), and add the row to
        `images`. `camera` is the camera's object id, `name` its name."""
        stem = f"{frame:06d}"
        image = f"cam/{name}/{stem}.jpg"
        (self.folder / image).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(self.folder / image, quality=JPEG_QUALITY)
        mask_path = None
        if mask is not None:
            mask_path = f"mask/{name}/{stem}.png"
            (self.folder / mask_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(mask).astype(np.uint16)).save(self.folder / mask_path)
        self.add("images", {"frame": frame, "camera": camera, "time": time, "image": image, "mask": mask_path})

    def finish(self, info: Optional[dict] = None) -> dict:
        """Write every table that has rows, and episode.json; return
        episode.json's contents."""
        tables = []
        for name, rows in self.rows.items():
            columns = self.schemas[name]
            key = columns.metadata[b"key"].decode().split(",")
            rows = sorted(rows, key=lambda r: tuple(-1 if r.get(k) is None else r[k] for k in key))
            table = pa.Table.from_pylist([{c: plain(r.get(c)) for c in columns.names} for r in rows], schema=columns)
            pq.write_table(table, self.folder / f"{name}.parquet", compression="zstd")
            tables.append(name)
        episode = {"format": FORMAT, **self.info, **(info or {}),
                   "frames": len(self.frames), "tables": tables}
        text_ = json.dumps(episode, indent=2, default=as_json)
        (self.folder / "episode.json").write_text(text_ + "\n")
        return json.loads(text_)


def plain(value):
    """numpy values as Python ones."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def as_json(value):
    out = plain(value)
    if out is value:
        raise TypeError(f"Can't write {type(value).__name__} to episode.json")
    return out
