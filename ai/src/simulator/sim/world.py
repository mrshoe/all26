"""`Scene` and `Scenario`, and building an episode's world from them.

A Scene is an environment (floor, walls, lights, static things) and is
written once and reused. A Scenario names its Scene, `Scenario[Gym]`, then
changes the Scene's objects, adds its own, and moves things in `step`.

Objects are kept as attributes (`self.floor = self.add(Floor(...))`), and
the framework names each one after its attribute: `floor`, `walls.0`. Those
names are what the logs, masks and visualizer call them.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from sim.geometry import quat_from_euler
from sim.objects import Background, Camera, Floor, Item, Light, Solid
from sim.randomness import Range, resolve

T = TypeVar("T", bound=Union[Item, list])


class _Holder:
    def add(self, item: T) -> T:
        """Put an object (or a list of them) in the world, and return it."""
        items = item if isinstance(item, list) else [item]
        for one in items:
            if not isinstance(one, Item):
                raise TypeError(f"add() takes objects like Sphere(...) or Camera(...), not {one!r}")
            self._items().append(one)
        return item

    def remove(self, item: Item) -> None:
        """Take an object out of the world, e.g. one of the Scene's."""
        for holder in (self, getattr(self, "scene", None)):
            if holder is not None and item in holder._items():
                holder._items().remove(item)
                return
        raise ValueError(f"{item!r} isn't in the world")

    def _items(self) -> list:
        if "_added" not in self.__dict__:
            self.__dict__["_added"] = []
        return self.__dict__["_added"]


class Scene(_Holder):
    """An environment: subclass it and add things in `setup`."""

    def setup(self, rng: np.random.Generator) -> None:
        pass


S = TypeVar("S", bound=Scene)


class Scenario(_Holder, Generic[S]):
    """What happens in a Scene: subclass `Scenario[MyScene]`, change the Scene
    and add things in `setup`, and move things in `step`."""

    scene: S
    # Seconds of simulated time per episode.
    duration: float = 5.0
    # Frames per second, for cameras and the logged tables.
    fps: int = 10
    # Physics steps per second.
    physics_hz: int = 240

    def setup(self, rng: np.random.Generator) -> None:
        pass

    def step(self, t: float, rng: np.random.Generator) -> None:
        """Called before every physics step, at time `t` in seconds."""

    @classmethod
    def scene_class(cls) -> type[Scene]:
        for klass in cls.__mro__:
            for base in getattr(klass, "__orig_bases__", ()):
                if typing.get_origin(base) is Scenario:
                    (scene,) = typing.get_args(base)
                    if isinstance(scene, type) and issubclass(scene, Scene):
                        return scene
        raise TypeError(f"Say which Scene {cls.__name__} uses: class {cls.__name__}(Scenario[MyScene])")


@dataclass
class World:
    """One episode's world, with every Range drawn."""

    scenario: Scenario
    seed: int
    items: list[Item]
    step_rng: np.random.Generator
    # Scripted changes from step() for the backend: (item, what, value).
    pending: list = field(default_factory=list)
    # describe() before the episode started: everything as drawn.
    drawn: dict = field(default_factory=dict)

    @property
    def scene(self) -> Scene:
        return self.scenario.scene

    def by_name(self, name: str) -> Item:
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(name)

    def of_type(self, kind: type) -> list:
        return [i for i in self.items if isinstance(i, kind)]

    @property
    def solids(self) -> list[Solid]:
        return self.of_type(Solid)

    @property
    def cameras(self) -> list[Camera]:
        return self.of_type(Camera)

    @property
    def lights(self) -> list[Light]:
        return self.of_type(Light)

    @property
    def background(self) -> Optional[Background]:
        found = self.of_type(Background)
        return found[0] if found else None

    def moving(self) -> list[Solid]:
        return [s for s in self.solids if not s.static]

    def labeled(self) -> list[Solid]:
        return [s for s in self.solids if s.label]

    # Called by objects once the episode has started (see Item.__setattr__).
    def changed(self, item: Item, key: str, value) -> None:
        if isinstance(item, Solid) and item.static:
            raise RuntimeError(f"{item.name} is static, so it can't move. Make it static=False.")
        if key == "rotation":
            item.__dict__["_quat"] = quat_from_euler(value)
        self.pending.append((item, key, value))

    def push(self, item: Item, impulse) -> None:
        self.pending.append((item, "push", impulse))

    def describe(self) -> dict:
        """Every object's settings, by name."""
        return {
            item.name: {"type": type(item).__name__, "id": item.id, "label": item.label, **item.settings()}
            for item in self.items
        }


def jsonable(value):
    """`value` with numpy types and tuples made plain, for JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


def build(scenario_class: type[Scenario], seed: int) -> World:
    """Run the Scene's and Scenario's setup with this seed and draw every
    Range. Three separate random streams come from the seed: one for setup's
    `rng`, one for Ranges and one for step's `rng`, so adding a Range doesn't
    change what setup draws, and so on."""
    setup_rng, range_rng, step_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    scene = scenario_class.scene_class()()
    scene.setup(setup_rng)
    scenario = scenario_class()
    scenario.scene = scene
    scenario.setup(setup_rng)

    items = scene._items() + scenario._items()
    if len(set(map(id, items))) != len(items):
        raise ValueError("The same object was added twice")
    name_items(scene, scenario, items)
    for number, item in enumerate(items, start=1):
        for f in dataclasses.fields(item):
            if f.init:
                object.__setattr__(item, f.name, resolve(getattr(item, f.name), range_rng))
        object.__setattr__(item, "id", number)
    world = World(scenario=scenario, seed=seed, items=items, step_rng=step_rng)
    check(world)
    world.drawn = world.describe()
    for item in items:
        item.__dict__["_world"] = world
    return world


def name_items(scene: Scene, scenario: Scenario, items: list[Item]) -> None:
    """Name each object after the attribute that holds it."""
    in_world = set(map(id, items))
    names: dict[int, str] = {}
    owner: dict[str, str] = {}
    for holder, where in ((scene, "Scene"), (scenario, "Scenario")):
        for attr, value in vars(holder).items():
            if attr.startswith("_") or attr == "scene":
                continue
            found = []
            if isinstance(value, Item):
                found = [(attr, value)]
            elif isinstance(value, (list, tuple)):
                found = [(f"{attr}.{i}", v) for i, v in enumerate(value) if isinstance(v, Item)]
            for name, item in found:
                if id(item) not in in_world or id(item) in names:
                    continue
                if name in owner:
                    raise ValueError(
                        f"Both the {owner[name]} and the {where} have an object called "
                        f"{name!r}. Rename one of them."
                    )
                names[id(item)], owner[name] = name, where
    counts: dict[str, int] = {}
    for item in items:
        if id(item) not in names:
            kind = type(item).__name__.lower()
            counts[kind] = counts.get(kind, 0) + 1
            names[id(item)] = f"{kind}.{counts[kind]}"
        object.__setattr__(item, "name", names[id(item)])


def check(world: World) -> None:
    """Catch mistakes before anything is simulated."""
    items = set(map(id, world.items))
    for light in world.lights:
        if light.kind not in Light.KINDS:
            raise ValueError(f"{light.name}: kind must be one of {', '.join(Light.KINDS)}")
    for floor in world.of_type(Floor):
        if floor.texture not in Floor.TEXTURES:
            raise ValueError(f"{floor.name}: texture must be one of {', '.join(Floor.TEXTURES)}")
    for cam in world.cameras:
        if isinstance(cam.look_at, Item) and id(cam.look_at) not in items:
            raise ValueError(f"{cam.name} looks at an object that isn't in the world")
        if not 1 <= cam.fov < 180:
            raise ValueError(f"{cam.name}: fov is in degrees, between 1 and 180")
    if len(world.of_type(Background)) > 1:
        raise ValueError("There can be only one Background")
    s = world.scenario
    if s.physics_hz % s.fps:
        raise ValueError(f"physics_hz ({s.physics_hz}) must be a multiple of fps ({s.fps})")
    for item in world.items:
        for f in dataclasses.fields(item):
            if isinstance(getattr(item, f.name), Range):
                raise AssertionError(f"{item.name}.{f.name} wasn't drawn")
