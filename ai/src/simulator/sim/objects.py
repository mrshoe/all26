"""What a Scene or Scenario can put in the world.

Units are SI: meters, kilograms and seconds, with angles in degrees. The
world's z axis points up. Any value can be a `Range`, drawn per episode.

Once the episode starts, an object's `position`, `rotation` and `velocity`
are live: reading them gives the simulation's current values, and setting
them in `Scenario.step` moves the object there.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union

import numpy as np

from sim.geometry import euler_from_quat, quat_from_euler

Number = Union[float, Any]  # a float, or a Range
Vec3 = Union[tuple, Any]
Color = Union[tuple, Any]  # sRGB, 0 to 1, or Range.color(...)

GRAY = (0.5, 0.5, 0.5)


@dataclass(eq=False, kw_only=True)
class Item:
    """Anything in the world."""

    position: Vec3 = (0.0, 0.0, 0.0)
    # Degrees about x, then y, then z. (A factory, so the class itself has no
    # `rotation` and __getattr__ can work it out; see there.)
    rotation: Vec3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    # What a detector should call it, e.g. "ball". Unlabeled things get no
    # boxes. Many objects can share a label.
    label: Optional[str] = None

    # Set by the framework.
    name: str = field(default="", init=False, repr=False)
    id: int = field(default=0, init=False, repr=False)

    LIVE: ClassVar[tuple[str, ...]] = ("position", "rotation")

    def __setattr__(self, key, value):
        world = self.__dict__.get("_world")
        if world is not None:
            if key not in self.LIVE:
                raise AttributeError(
                    f"{self.name}.{key} can't change once the episode has started. "
                    f"Set it in setup(), or change {', '.join(self.LIVE)} instead."
                )
            world.changed(self, key, value)
        object.__setattr__(self, key, value)

    def __getattr__(self, key):
        # Only called for attributes the object doesn't have. While physics
        # moves an object, it keeps just the quaternion each step, and the
        # rotation in degrees is worked out the first time it's read.
        if key == "rotation" and self.__dict__.get("_quat") is not None:
            self.__dict__["rotation"] = euler_from_quat(self.__dict__["_quat"])
            return self.__dict__["rotation"]
        raise AttributeError(f"{type(self).__name__} has no attribute {key!r}")

    @property
    def quat(self) -> np.ndarray:
        """The rotation as a (w, x, y, z) quaternion."""
        live = self.__dict__.get("_quat")
        return live if live is not None else quat_from_euler(self.rotation)

    def settings(self) -> dict:
        """The object's settings, as recorded with the episode."""
        out = {}
        for f in dataclasses.fields(self):
            if f.init and f.name != "label":
                value = getattr(self, f.name)
                out[f.name] = value.name if isinstance(value, Item) else value
        return out


@dataclass(eq=False, kw_only=True)
class Solid(Item):
    """Something with a shape that physics and cameras see."""

    color: Color = GRAY
    # 0 is matte, 1 is mirror-smooth.
    roughness: Number = 0.5
    # Static things never move; the others fall, bounce and get pushed.
    static: bool = True
    mass: Number = 1.0
    velocity: Vec3 = (0.0, 0.0, 0.0)
    # 0 doesn't bounce at all; 0.9 bounces back 0.81 as high as it fell
    # (the coefficient of restitution). Above 0.95 counts as 0.95.
    bounciness: Number = 0.3
    friction: Number = 0.6

    LIVE: ClassVar[tuple[str, ...]] = ("position", "rotation", "velocity")

    def push(self, impulse: Vec3) -> None:
        """Change the velocity by impulse / mass, in N·s, e.g. a kick."""
        world = self.__dict__.get("_world")
        if world is None:
            raise RuntimeError("push() only works once the episode has started, in step()")
        if self.static:
            raise RuntimeError(f"{self.name} is static, so it can't be pushed")
        world.push(self, impulse)


@dataclass(eq=False, kw_only=True)
class Sphere(Solid):
    radius: Number = 0.1
    static: bool = False
    mass: Number = 0.2
    bounciness: Number = 0.6


@dataclass(eq=False, kw_only=True)
class Box(Solid):
    # Full lengths along x, y and z.
    size: Vec3 = (1.0, 1.0, 1.0)


@dataclass(eq=False, kw_only=True)
class Floor(Solid):
    """A flat floor at z = 0, centered on the origin."""

    # Full lengths along x and y.
    size: Vec3 = (10.0, 10.0)
    texture: Any = "plain"

    TEXTURES: ClassVar[tuple[str, ...]] = (
        "plain", "wood", "wood_dark", "wood_red", "concrete", "carpet", "checker", "tile",
    )
    WOOD: ClassVar[tuple[str, ...]] = ("wood", "wood_dark", "wood_red")


@dataclass(eq=False, kw_only=True)
class Wall(Solid):
    """A wall from `start` to `end` (x, y), standing on the floor."""

    start: Vec3 = (0.0, 0.0)
    end: Vec3 = (1.0, 0.0)
    height: Number = 3.0
    thickness: Number = 0.2

    def box(self) -> tuple[np.ndarray, np.ndarray, float]:
        """The wall as a box: its center, full size (x, y, z) and turn about
        z in degrees. It runs past each end by half its thickness, so walls
        that share a corner close it."""
        start, end = np.array(self.start[:2], float), np.array(self.end[:2], float)
        along = end - start
        length = float(np.linalg.norm(along)) + self.thickness
        center = np.array([*(start + end) / 2, self.height / 2])
        yaw = float(np.degrees(np.arctan2(along[1], along[0])))
        return center, np.array([length, self.thickness, self.height]), yaw

    @staticmethod
    def around(floor: Floor, height: Number = 3.0, **kwargs) -> list["Wall"]:
        """Four walls around the edges of a floor. Other settings (color, ...)
        apply to each wall, and a Range is drawn for each wall separately."""
        sx, sy = floor.size
        if not all(isinstance(v, (int, float)) for v in (sx, sy)):
            raise ValueError("Wall.around needs a floor whose size isn't a Range")
        x, y = sx / 2, sy / 2
        corners = [(-x, -y), (x, -y), (x, y), (-x, y)]
        return [
            Wall(start=corners[i], end=corners[(i + 1) % 4], height=height, **kwargs)
            for i in range(4)
        ]


@dataclass(eq=False, kw_only=True)
class Light(Item):
    """A light. `kind` is "sun" (parallel rays from far away), "area" (a glowing
    rectangle, like a ceiling panel), "point" or "spot"."""

    kind: str = "sun"
    # Sun: watts per square meter (1 to 10 is daylight). Others: watts.
    intensity: Number = 3.0
    color: Color = (1.0, 1.0, 1.0)
    # Which way the light shines, for sun, area and spot lights.
    direction: Vec3 = (0.0, 0.0, -1.0)
    # Area lights: full lengths (x, y). Sun: the sun's angular size in degrees,
    # for softer shadows.
    size: Any = (1.0, 1.0)

    KINDS: ClassVar[tuple[str, ...]] = ("sun", "area", "point", "spot")


@dataclass(eq=False, kw_only=True)
class Camera(Item):
    """A camera. Point it with `look_at` (an object, which it follows, or a
    point) or with `rotation`; it looks along its own -z axis with +y up."""

    resolution: tuple = (640, 480)
    # Horizontal field of view, in degrees.
    fov: Number = 70.0
    look_at: Any = None

    LIVE: ClassVar[tuple[str, ...]] = ("position", "rotation", "look_at")


@dataclass(eq=False, kw_only=True)
class Background(Item):
    """What's seen where nothing else is, and the light it gives (sky light
    outdoors). Indoors, walls hide it."""

    color: Color = (0.6, 0.7, 0.85)
    strength: Number = 1.0
