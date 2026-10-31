"""`Logger`: writes an episode's data to a folder.

    episode.json            what ran: scenario, seed, everything drawn,
                            cameras and their intrinsics, signal names
    cam/<camera>/000000.jpg what each camera saw, one file per frame
    mask/<camera>/000000.png which object each pixel shows (16-bit Item ids;
                            episode.json's objects map ids to names)
    signals.parquet         one row per frame: time, each moving object's
                            and camera's pose, and each labeled object's box
                            in each camera

Signal columns are named <object>.<what>, e.g. ball.z, front.qw, and
front.ball.x0 for the ball's box in the front camera. Boxes are in pixels,
with (x0, y0) the top-left corner and (x1, y1) one past the bottom-right;
they're empty (NaN) in frames where the object isn't seen. <camera>.<object>
.pixels counts the pixels it covers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from sim.backends import Picture
from sim.geometry import intrinsics
from sim.world import World, jsonable

FORMAT = 1
JPEG_QUALITY = 90


def boxes(ids: np.ndarray, items) -> dict[str, tuple]:
    """Each labeled item's tight box in a mask, and its pixel count."""
    out = {}
    for item in items:
        ys, xs = np.nonzero(ids == item.id)
        if len(xs):
            out[item.name] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, len(xs))
        else:
            out[item.name] = (math.nan,) * 4 + (0,)
    return out


class Logger:
    def __init__(self, folder: Path, world: World, info: dict, render: bool):
        self.folder, self.world, self.info, self.render = Path(folder), world, info, render
        self.folder.mkdir(parents=True, exist_ok=True)
        self.columns: dict[str, list] = {}

    def frame(self, index: int, t: float, cameras: dict, pictures: dict[str, Picture]) -> None:
        """Log one frame. `cameras` maps each camera's name to its (position,
        quat), and `pictures` its name to what it saw."""
        row = {"frame": index, "time": t}
        for item in self.world.moving():
            row.update(pose_columns(item.name, item.position, item.quat))
        for name, (position, quat) in cameras.items():
            row.update(pose_columns(name, position, quat))
        for name, picture in pictures.items():
            for obj, (x0, y0, x1, y1, n) in boxes(picture.ids, self.world.labeled()).items():
                row.update({f"{name}.{obj}.x0": x0, f"{name}.{obj}.y0": y0,
                            f"{name}.{obj}.x1": x1, f"{name}.{obj}.y1": y1,
                            f"{name}.{obj}.pixels": n})
            stem = f"{index:06d}"
            for kind, image in (("cam", Image.fromarray(picture.rgb)),
                                ("mask", Image.fromarray(picture.ids.astype(np.uint16)))):
                path = self.folder / kind / name / f"{stem}.{'jpg' if kind == 'cam' else 'png'}"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, quality=JPEG_QUALITY) if kind == "cam" else image.save(path)
        for key, value in row.items():
            self.columns.setdefault(key, []).append(value)

    def finish(self) -> dict:
        """Write signals.parquet and episode.json, and return episode.json's
        contents."""
        table = pa.table({k: pa.array(v, type=pa.int32() if k == "frame" else pa.float64())
                          for k, v in self.columns.items()})
        pq.write_table(table, self.folder / "signals.parquet")
        world, scenario = self.world, self.world.scenario
        episode = {
            "format": FORMAT,
            **self.info,
            "seed": world.seed,
            "fps": scenario.fps,
            "duration": scenario.duration,
            "physics_hz": scenario.physics_hz,
            "frames": len(self.columns.get("frame", [])),
            "cameras": {
                cam.name: {"fov": cam.fov, **intrinsics(cam.resolution, cam.fov)}
                for cam in world.cameras
            } if self.render else {},
            "labels": {item.name: item.label for item in world.labeled()},
            "objects": world.drawn,
            "signals": list(self.columns),
        }
        (self.folder / "episode.json").write_text(json.dumps(jsonable(episode), indent=2) + "\n")
        return episode


def pose_columns(name: str, position, quat) -> dict:
    x, y, z = map(float, position)
    qw, qx, qy, qz = map(float, quat)
    return {f"{name}.x": x, f"{name}.y": y, f"{name}.z": z,
            f"{name}.qw": qw, f"{name}.qx": qx, f"{name}.qy": qy, f"{name}.qz": qz}
