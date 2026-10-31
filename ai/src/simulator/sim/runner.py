"""The episode loop, which the framework owns:

    build the world from the seed
    for each frame:
        call Scenario.step and advance physics, physics_hz / fps times
        point each camera, and render what it sees
        log the frame
    write the logs, then make previews
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from sim import backends
from sim.geometry import intrinsics, project, quat_from_euler, quat_looking_at
from sim.objects import Camera, Item
from sim.world import Scenario, World, build


def load(target: str) -> tuple[type[Scenario], str]:
    """The Scenario in a file, `scenarios/x.py` or `scenarios/x.py:Name`,
    and how to refer to it ("scenarios/x.py:Name")."""
    path, _, name = target.partition(":")
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"sim: there's no file {path}")
    # So the scenario can import scenes/ and anything else here.
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    spec = importlib.util.spec_from_file_location(f"scenario_{file.stem}", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = [
        v for v in vars(module).values()
        if inspect.isclass(v) and issubclass(v, Scenario) and v.__module__ == module.__name__
    ]
    if name:
        found = [v for v in found if v.__name__ == name]
    if len(found) != 1:
        names = ", ".join(v.__name__ for v in found) or "none"
        raise SystemExit(f"sim: expected one Scenario in {target}, found {names}. Use {path}:Name.")
    return found[0], f"{file.as_posix()}:{found[0].__name__}"


def camera_pose(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(cam.position, dtype=float)
    target = cam.look_at
    if isinstance(target, Item):
        return position, quat_looking_at(position, target.position)
    if target is not None:
        return position, quat_looking_at(position, target)
    return position, cam.quat


def apply_changes(world: World, backend) -> None:
    """Pass step()'s changes on to physics."""
    for item, what, value in world.pending:
        if isinstance(item, Camera):
            continue  # cameras aren't in physics
        if what == "position":
            backend.set_pose(item, value, item.quat)
        elif what == "rotation":
            backend.set_pose(item, item.position, quat_from_euler(value))
        elif what == "velocity":
            backend.set_velocity(item, value)
        elif what == "push":
            backend.push(item, value)
    world.pending.clear()


def read_poses(world: World, backend) -> None:
    """Make moving objects' position, rotation and velocity what physics says."""
    poses = backend.poses()
    for item in world.moving():
        pose = poses[item.id]
        d = item.__dict__
        d["position"] = tuple(float(v) for v in pose.position)
        d["_quat"] = pose.quat
        d.pop("rotation", None)  # worked out if read (Item.__getattr__)
        d["velocity"] = tuple(float(v) for v in pose.velocity)


@dataclass
class Episode:
    world: World
    frames: int = 0
    # For `sim check`: per camera, per labeled object, in how many frames
    # its center is inside the picture.
    in_view: dict = field(default_factory=dict)
    seconds: float = 0.0


def run(scenario_class: type[Scenario], seed: int, backend_name: str,
        logger_factory=None, progress: bool = False) -> Episode:
    started = time.time()
    world = build(scenario_class, seed)
    backend = backends.get(backend_name)
    backend.build(world)
    logger = logger_factory(world) if logger_factory else None
    scenario = world.scenario
    steps = scenario.physics_hz // scenario.fps
    dt = 1 / scenario.physics_hz
    frames = round(scenario.duration * scenario.fps)
    episode = Episode(world=world)
    for cam in world.cameras:
        episode.in_view[cam.name] = {item.name: 0 for item in world.labeled()}
    read_poses(world, backend)
    try:
        for frame in range(frames):
            if frame:
                for s in range(steps):
                    t = ((frame - 1) * steps + s) * dt
                    scenario.step(t, world.step_rng)
                    apply_changes(world, backend)
                    backend.step(dt)
                    read_poses(world, backend)
            t = frame / scenario.fps
            poses, pictures = {}, {}
            for cam in world.cameras:
                position, quat = camera_pose(cam)
                poses[cam.name] = (position, quat)
                k = intrinsics(cam.resolution, cam.fov)
                for item in world.labeled():
                    p = project(item.position, position, quat, k)
                    if p and 0 <= p[0] < k["width"] and 0 <= p[1] < k["height"]:
                        episode.in_view[cam.name][item.name] += 1
                picture = backend.render(cam, position, quat)
                if picture is not None:
                    pictures[cam.name] = picture
            if logger:
                logger.frame(frame, t, poses, pictures)
            episode.frames += 1
            if progress and pictures:
                print(f"sim: frame {frame + 1}/{frames} ({time.time() - started:.0f}s)", flush=True)
    finally:
        backend.close()
    episode.seconds = time.time() - started
    return episode
