"""`SimLogger`: turns the simulation into the shared `Logger`'s tables
(ai/src/shared/logger.py has the schemas).

    objects  every object: its id (the one in the masks), name, type, label,
             whether it's static, and what was drawn for it
    cameras  each camera's resolution and intrinsics
    sim      each frame's time, and how long step(), physics, rendering and
             logging took
    poses    moving objects, cameras and lights every frame; static things
             once, at frame 0. Velocities come from physics for moving
             objects, and from the change since the last frame for cameras
             and lights.
    images   each camera's picture and mask
    boxes    each labeled object's box in each camera's picture, from the
             mask, in frames where it's seen; and its full box, from its
             whole shape, as if nothing hid it (sim/labels.py)
    keypoints
             every object's named points (Solid.keypoints) in each camera's
             picture, seen or hidden, from the depth image (sim/labels.py)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from shared.logger import Logger
from sim import labels
from sim.backends import Picture
from sim.geometry import intrinsics, quat_from_euler, quat_looking_at, spin_between
from sim.objects import Background, Camera, Item, Light, Solid, Wall
from sim.world import World, jsonable


def boxes(ids: np.ndarray, items) -> dict[Item, tuple]:
    """Each labeled item's tight box in a mask, and its pixel count, for the
    items it shows."""
    out = {}
    for item in items:
        ys, xs = np.nonzero(ids == item.id)
        if len(xs):
            out[item] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, len(xs))
    return out


def is_static(item: Item) -> bool:
    """Logged once, at frame 0. Cameras and lights can move in step()."""
    return isinstance(item, Background) or (isinstance(item, Solid) and item.static)


def static_pose(item: Item) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(item, Wall):
        center, _, yaw = item.box()
        return center, quat_from_euler((0, 0, yaw))
    return np.asarray(item.position, float), item.quat


def light_pose(light: Light) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(light.position, float)
    if light.kind == "point":
        return position, light.quat
    return position, quat_looking_at(position, position + np.asarray(light.direction, float))


class SimLogger:
    def __init__(self, folder: Path, world: World, info: dict, render: bool):
        self.world, self.info, self.render = world, info, render
        self.log = Logger(folder)
        self.last: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}  # id -> (t, position, quat)
        # Worked out once: each labeled object's shape, and each object's keypoints.
        self.shapes = {item.id: labels.shape(item) for item in world.labeled()} if render else {}
        self.points = {item.id: found for item in world.solids if (found := labels.keypoints_of(item))}
        for item in world.items:
            settings = {k: v for k, v in world.drawn[item.name].items() if k not in ("type", "id", "label")}
            self.log.add("objects", {
                "object": item.id, "name": item.name, "type": type(item).__name__,
                "label": getattr(item, "label", None) or None, "static": is_static(item),
                "settings": json.dumps(jsonable(settings)),
            })
        if render:
            for cam in world.cameras:
                k = intrinsics(cam.resolution, cam.fov)
                self.log.add("cameras", {"object": cam.id, "width": k["width"], "height": k["height"],
                                         "fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"],
                                         "fov": float(cam.fov)})

    def pose(self, frame: int, t: float, item: Item, position, quat, velocity=None, spin=None) -> None:
        """A pose row. Without velocities, they come from the last frame's pose."""
        if velocity is None:
            last = self.last.get(item.id)
            if last and t > last[0]:
                dt = t - last[0]
                velocity = (np.asarray(position) - last[1]) / dt
                spin = spin_between(last[2], quat, dt)
            else:
                velocity = spin = np.zeros(3)
            self.last[item.id] = (t, np.asarray(position, float), np.asarray(quat, float))
        x, y, z = map(float, position)
        qw, qx, qy, qz = map(float, quat)
        vx, vy, vz = map(float, velocity)
        wx, wy, wz = map(float, spin)
        self.log.add("poses", {"frame": frame, "object": item.id, "x": x, "y": y, "z": z,
                               "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                               "vx": vx, "vy": vy, "vz": vz, "wx": wx, "wy": wy, "wz": wz})

    def frame(self, index: int, t: float, cameras: dict, pictures: dict[str, Picture], timing: dict) -> None:
        """Log one frame. `cameras` maps each camera's name to its (position,
        quat), `pictures` its name to what it saw, and `timing` says how long
        step(), physics and rendering took."""
        started = time.perf_counter()
        world = self.world
        if index == 0:
            for item in world.items:
                if is_static(item):
                    position, quat = static_pose(item)
                    self.pose(0, t, item, position, quat, np.zeros(3), np.zeros(3))
        for item in world.moving():
            self.pose(index, t, item, item.position, item.quat, item.velocity,
                      item.__dict__.get("_spin", np.zeros(3)))
        for light in world.lights:
            self.pose(index, t, light, *light_pose(light))
        named = {cam.name: cam for cam in world.cameras}
        for name, (position, quat) in cameras.items():
            self.pose(index, t, named[name], position, quat)
        for name, picture in pictures.items():
            cam = named[name]
            self.log.image(index, cam.id, name, t, picture.rgb, picture.ids)
            view = labels.View(*cameras[name], intrinsics(cam.resolution, cam.fov))
            for item, (x0, y0, x1, y1, n) in boxes(picture.ids, world.labeled()).items():
                row = {"frame": index, "camera": cam.id, "object": item.id, "label": item.label,
                       "x0": x0, "y0": y0, "x1": x1, "y1": y1, "pixels": n, "source": "truth"}
                full = labels.full_box(view, item, self.shapes[item.id])
                if full:
                    row.update(zip(("full_x0", "full_y0", "full_x1", "full_y1", "full_pixels"), full))
                self.log.add("boxes", row)
            if picture.depth is not None:
                self.keypoints(index, cam, view, picture)
        self.log.add("sim", {"frame": index, "time": t, **timing, "log_s": time.perf_counter() - started})

    def keypoints(self, index: int, cam: Camera, view: labels.View, picture: Picture) -> None:
        for item in self.world.solids:
            if item.id not in self.points:
                continue
            for p in labels.keypoints(view, item, self.points[item.id], picture.ids, picture.depth):
                self.log.add("keypoints", {
                    "frame": index, "camera": cam.id, "object": item.id, "label": p.label, "index": p.index,
                    "x": p.x, "y": p.y, "depth": p.depth, "in_frame": p.in_frame, "visible": p.visible,
                    "occluded_by": p.occluded_by, "source": "truth"})

    def finish(self) -> dict:
        """Write the episode's files; return episode.json's contents."""
        scenario = self.world.scenario
        return self.log.finish(jsonable({
            **self.info,
            "seed": self.world.seed,
            "fps": scenario.fps,
            "duration": scenario.duration,
            "physics_hz": scenario.physics_hz,
            # Each Group's settings as drawn; its objects are in the objects table.
            "groups": self.world.groups,
        }))
