"""`SimLogger`: turns the simulation into what the shared `Logger`
(ai/src/shared/logger.py) writes, frame by frame.

    signals    each moving object's and camera's pose (<name>.x/y/z and
               <name>.qw/qx/qy/qz), and each labeled object's box in each
               camera (<camera>.<object>.x0/y0/x1/y1/pixels), from the masks
    images     each camera's picture and object-id mask
    episode    scenario, Scene, backend, seed, timing, cameras and their
               intrinsics, labels, and every object's drawn settings
               (`objects`, which also maps mask ids to names)

Boxes are in pixels, with (x0, y0) the top-left corner and (x1, y1) one past
the bottom-right; they're NaN in frames where the object isn't seen.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from shared.logger import Logger
from sim.backends import Picture
from sim.geometry import intrinsics
from sim.world import World, jsonable


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


class SimLogger:
    def __init__(self, folder: Path, world: World, info: dict, render: bool):
        self.world, self.info, self.render = world, info, render
        self.log = Logger(folder)

    def frame(self, index: int, t: float, cameras: dict, pictures: dict[str, Picture]) -> None:
        """Log one frame. `cameras` maps each camera's name to its (position,
        quat), and `pictures` its name to what it saw."""
        signals = {}
        for item in self.world.moving():
            signals.update(pose_columns(item.name, item.position, item.quat))
        for name, (position, quat) in cameras.items():
            signals.update(pose_columns(name, position, quat))
        for name, picture in pictures.items():
            for obj, (x0, y0, x1, y1, n) in boxes(picture.ids, self.world.labeled()).items():
                signals.update({f"{name}.{obj}.x0": x0, f"{name}.{obj}.y0": y0,
                                f"{name}.{obj}.x1": x1, f"{name}.{obj}.y1": y1,
                                f"{name}.{obj}.pixels": n})
        self.log.frame(index, t, signals,
                       images={name: p.rgb for name, p in pictures.items()},
                       masks={name: p.ids for name, p in pictures.items()})

    def finish(self) -> dict:
        """Write the episode's files; return episode.json's contents."""
        world, scenario = self.world, self.world.scenario
        return self.log.finish(jsonable({
            **self.info,
            "seed": world.seed,
            "fps": scenario.fps,
            "duration": scenario.duration,
            "physics_hz": scenario.physics_hz,
            "cameras": {
                cam.name: {"fov": cam.fov, **intrinsics(cam.resolution, cam.fov)}
                for cam in world.cameras
            } if self.render else {},
            "labels": {item.name: item.label for item in world.labeled()},
            "objects": world.drawn,
        }))


def pose_columns(name: str, position, quat) -> dict:
    x, y, z = map(float, position)
    qw, qx, qy, qz = map(float, quat)
    return {f"{name}.x": x, f"{name}.y": y, f"{name}.z": z,
            f"{name}.qw": qw, f"{name}.qx": qx, f"{name}.qy": qy, f"{name}.qz": qz}
