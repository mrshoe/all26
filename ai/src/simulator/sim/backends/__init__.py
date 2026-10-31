"""Backends turn a World into a simulation: physics, and pictures.

Scenes and Scenarios never touch a backend, so swapping one (Blender for
Isaac Sim, say) changes how things look, not anyone's code.

- `physics`: MuJoCo only, no pictures. `python -m sim check` uses it.
- `blender`: MuJoCo for physics, Blender (Cycles, on the CPU) for pictures.
- `isaac` (later): Isaac Sim's physics and rendering, on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from sim.objects import Camera, Solid
from sim.world import World


@dataclass
class Pose:
    position: np.ndarray  # (x, y, z)
    quat: np.ndarray  # (w, x, y, z)
    velocity: np.ndarray  # (x, y, z)


@dataclass
class Picture:
    rgb: np.ndarray  # height x width x 3, uint8
    # height x width, uint16: each pixel's object id (Item.id), 0 for none.
    ids: np.ndarray


class Backend(Protocol):
    name: str

    def build(self, world: World) -> None:
        """Create everything in the world."""

    def step(self, dt: float) -> None:
        """Advance physics by dt seconds."""

    def poses(self) -> dict[int, Pose]:
        """Every moving object's pose, by Item.id."""

    def set_pose(self, item: Solid, position, quat) -> None: ...

    def set_velocity(self, item: Solid, velocity) -> None: ...

    def push(self, item: Solid, impulse) -> None: ...

    def render(self, camera: Camera, position, quat) -> Optional[Picture]:
        """What the camera sees from this pose, or None if this backend
        doesn't render."""

    def close(self) -> None: ...


def get(name: str) -> Backend:
    if name == "physics":
        from sim.backends.physics import Physics

        return Physics()
    if name == "blender":
        from sim.backends.blender import Blender

        return Blender()
    raise ValueError(f"Unknown backend {name!r}: use physics or blender")
