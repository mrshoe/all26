"""`Scene` and `Scenario`, and building an episode's world from them.

A Scene is an environment (floor, walls, lights, static things) and is
written once and reused. A Scenario names its Scene, `Scenario[Gym]`, then
changes the Scene's objects, adds its own, and moves things in `step`.

Objects are kept as attributes (`self.floor = self.add(Floor(...))`), and
the framework names each one after its attribute: `floor`, `walls.0`. Those
names are what the logs, masks and visualizer call them.

A Group is objects that go together, e.g. an FRC field, written once and
added to any Scene or Scenario. Its objects are placed relative to it and
named under it: `field.hub`.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar, Union

import numpy as np
from scipy.spatial.transform import Rotation

from sim import assets
from sim.geometry import quat_from_euler
from sim.objects import Background, Camera, Floor, GamePiece, Item, Light, Prop, Solid, Vec3, Wall
from sim.randomness import Range, resolve

T = TypeVar("T", bound=Union[Item, "Group", list])


class _Holder:
    def add(self, item: T) -> T:
        """Put an object, a Group, or a list of them in the world, and
        return it."""
        items = item if isinstance(item, list) else [item]
        for one in items:
            if isinstance(one, Group):
                self._add_group(one)
                continue
            if not isinstance(one, Item):
                raise TypeError(f"add() takes objects like Sphere(...) or Camera(...), not {one!r}")
            if isinstance(self, Group) and isinstance(one, (Floor, Background)):
                raise TypeError(f"A {type(one).__name__} can't be in a Group. Add it to the Scene instead.")
            self._items().append(one)
        return item

    def _add_group(self, group: Group) -> None:
        if group.__dict__.get("_built"):
            raise ValueError(f"{group!r} was already added")
        rng, range_rng = self.__dict__.get("_rng"), self.__dict__.get("_range_rng")
        if rng is None:
            raise RuntimeError("Groups can only be added in setup(), or in another Group's build()")
        group.__dict__["_rng"], group.__dict__["_range_rng"] = rng, range_rng
        # Its settings are drawn now, so build() can use them.
        for f in dataclasses.fields(group):
            if f.init:
                object.__setattr__(group, f.name, resolve(getattr(group, f.name), range_rng))
        group.build(rng)
        group.__dict__["_built"] = True
        for one in group._items():
            # Innermost Group first: that's the order they're placed in.
            one.__dict__.setdefault("_groups", []).append(group)
            self._items().append(one)
        self._groups().append(group)

    def remove(self, item: Union[Item, Group]) -> None:
        """Take an object or a Group out of the world, e.g. one of the
        Scene's."""
        holders = [h for h in (self, getattr(self, "scene", None)) if h is not None]
        if isinstance(item, Group):
            for holder in holders + [g for h in holders for g in all_groups(h)]:
                kept = [g for g in holder._groups() if g is not item]
                if len(kept) < len(holder._groups()):
                    holder._groups()[:] = kept
                    gone = set(map(id, item._items()))
                    for h in holders:
                        h._items()[:] = [one for one in h._items() if id(one) not in gone]
                    return
            raise ValueError(f"{item!r} isn't in the world")
        for holder in holders:
            if item in holder._items():
                holder._items().remove(item)
                return
        raise ValueError(f"{item!r} isn't in the world")

    def _items(self) -> list:
        if "_added" not in self.__dict__:
            self.__dict__["_added"] = []
        return self.__dict__["_added"]

    def _groups(self) -> list:
        if "_added_groups" not in self.__dict__:
            self.__dict__["_added_groups"] = []
        return self.__dict__["_added_groups"]


@dataclass(eq=False, kw_only=True)
class Group(_Holder):
    """Objects that go together, e.g. a field: subclass it, add its objects
    in `build`, and add it to any Scene or Scenario like an object.

        @dataclass(kw_only=True)
        class Goal(Group):
            width: Number = 2.0

            def build(self, rng):
                self.posts = self.add([Box(size=(0.1, 0.1, 1), position=(x, 0, 0.5))
                                       for x in (-self.width / 2, self.width / 2)])

        self.goal = self.add(Goal(position=(3, 0, 0), rotation=(0, 0, Range(0, 360))))

    Its objects' positions and rotations are relative to the Group's own.
    Its settings are drawn when it's added, so `build` sees numbers, not
    Ranges; its position and rotation can still be changed until the episode
    starts. Once the episode starts, its objects' positions are in the world's
    coordinates, like everything else's. Floors and Backgrounds belong to a
    Scene, so they can't be in a Group.
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    # Degrees about x, then y, then z.
    rotation: Vec3 = (0.0, 0.0, 0.0)
    name: str = field(default="", init=False, repr=False)

    def build(self, rng: np.random.Generator) -> None:
        """Add the Group's objects, with `self.add`. `rng` is setup's."""

    def settings(self) -> dict:
        """The Group's settings, as recorded with the episode."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}


def all_groups(holder: _Holder) -> list[Group]:
    """The Groups added to a holder, and the Groups in those, in order."""
    found = []
    for group in holder._groups():
        found += [group, *all_groups(group)]
    return found


def place(item: Item, group: Group) -> None:
    """Move an object from its Group's coordinates into the Group's parent's,
    once every Range is drawn."""
    turn = Rotation.from_euler("xyz", group.rotation, degrees=True)
    offset = np.asarray(group.position, dtype=float)

    def point(v) -> tuple:
        return tuple(float(x) for x in offset + turn.apply(np.asarray(v, dtype=float)))

    def set_(key, value):
        object.__setattr__(item, key, value)

    if isinstance(item, Wall):
        # Walls stand on the floor, so a Group with them can only turn about z.
        if not np.allclose(turn.apply([0, 0, 1]), [0, 0, 1], atol=1e-6) or abs(offset[2]) > 1e-9:
            raise ValueError(f"{item.name}: a Group with walls can only turn about z and must stay "
                             f"on the floor, but {group.name} is at {group.position}, turned {group.rotation}")
        set_("start", point((*item.start[:2], 0))[:2])
        set_("end", point((*item.end[:2], 0))[:2])
    else:
        set_("position", point(item.position))
        local = Rotation.from_euler("xyz", item.rotation, degrees=True)
        set_("rotation", tuple(float(a) for a in (turn * local).as_euler("xyz", degrees=True)))
    if isinstance(item, Solid):
        set_("velocity", tuple(float(x) for x in turn.apply(np.asarray(item.velocity, dtype=float))))
    if isinstance(item, Light):
        set_("direction", tuple(float(x) for x in turn.apply(np.asarray(item.direction, dtype=float))))
    if isinstance(item, Camera) and item.look_at is not None and not isinstance(item.look_at, Item):
        set_("look_at", point(item.look_at))


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
    # Each Group's settings as drawn, by name.
    groups: dict = field(default_factory=dict)

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
    scene.__dict__.update(_rng=setup_rng, _range_rng=range_rng)
    scene.setup(setup_rng)
    scenario = scenario_class()
    scenario.scene = scene
    scenario.__dict__.update(_rng=setup_rng, _range_rng=range_rng)
    scenario.setup(setup_rng)

    items = scene._items() + scenario._items()
    if len(set(map(id, items))) != len(items):
        raise ValueError("The same object was added twice")
    groups = all_groups(scene) + all_groups(scenario)
    name_items(scene, scenario, items, groups)
    for number, item in enumerate(items, start=1):
        for f in dataclasses.fields(item):
            if f.init:
                object.__setattr__(item, f.name, resolve(getattr(item, f.name), range_rng))
        object.__setattr__(item, "id", number)
    for group in groups:  # e.g. a position the Scenario changed after adding it
        for f in dataclasses.fields(group):
            if f.init:
                object.__setattr__(group, f.name, resolve(getattr(group, f.name), range_rng))
    present = set(map(id, groups))
    for item in items:
        for group in item.__dict__.get("_groups", []):
            if id(group) in present:
                place(item, group)
    world = World(scenario=scenario, seed=seed, items=items, step_rng=step_rng)
    check(world)
    world.drawn = world.describe()
    world.groups = {g.name: {"type": type(g).__name__, **g.settings()} for g in groups}
    for item in items:
        item.__dict__["_world"] = world
    return world


def name_items(scene: Scene, scenario: Scenario, items: list[Item], groups: list[Group] = ()) -> None:
    """Name each object after the attribute that holds it, and each Group's
    objects under the Group's name: `field.hub`."""
    in_world = set(map(id, items))
    in_groups = set(map(id, groups))
    names: dict[int, str] = {}
    owner: dict[str, str] = {}
    counts: dict[str, int] = {}

    def visit(holder, prefix: str, where: str) -> None:
        for attr, value in vars(holder).items():
            if attr.startswith("_") or attr == "scene":
                continue
            found = []
            if isinstance(value, (Item, Group)):
                found = [(attr, value)]
            elif isinstance(value, (list, tuple)):
                found = [(f"{attr}.{i}", v) for i, v in enumerate(value) if isinstance(v, (Item, Group))]
            for name, one in found:
                full = prefix + name
                if isinstance(one, Group):
                    if id(one) in in_groups and not one.name:
                        object.__setattr__(one, "name", full)
                        visit(one, full + ".", where)
                    continue
                if id(one) not in in_world or id(one) in names:
                    continue
                if full in owner:
                    raise ValueError(
                        f"Both the {owner[full]} and the {where} have an object called "
                        f"{full!r}. Rename one of them."
                    )
                names[id(one)], owner[full] = full, where

    def numbered(prefix: str, kind: str) -> str:
        counts[prefix + kind] = counts.get(prefix + kind, 0) + 1
        return f"{prefix}{kind}.{counts[prefix + kind]}"

    for group in groups:
        object.__setattr__(group, "name", "")
    visit(scene, "", "Scene")
    visit(scenario, "", "Scenario")
    # Groups not kept in an attribute, outermost first, so names nest.
    for group in groups:
        if not group.name:
            parent = next((g for g in groups if any(c is group for c in g._groups())), None)
            object.__setattr__(group, "name", numbered(parent.name + "." if parent else "", type(group).__name__.lower()))
            visit(group, group.name + ".", "Scene")
    for item in items:
        if id(item) not in names:
            inner = item.__dict__.get("_groups", [])
            prefix = inner[0].name + "." if inner else ""
            names[id(item)] = numbered(prefix, type(item).__name__.lower())
        object.__setattr__(item, "name", names[id(item)])


def check_keypoints(item: Solid) -> None:
    points = item.keypoints
    if points is None:
        return
    example = 'keypoints={"corner": [(x, y, z), ...]}'
    if not isinstance(points, dict):
        raise ValueError(f"{item.name}: keypoints are named lists of points, like {example}")
    for name, found in points.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{item.name}: name each class of keypoints, like {example}")
        shape = np.shape(found)
        if len(shape) != 2 or shape[1] != 3 or not shape[0]:
            raise ValueError(f"{item.name}: keypoints {name!r} should be a list of (x, y, z) points, "
                             f"like {example}")


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
    for prop in world.of_type(Prop):
        try:
            assets.info(prop.asset)
        except KeyError as e:
            raise ValueError(f"{prop.name}: {e.args[0]}") from None
        if not prop.static and not isinstance(prop, GamePiece):
            raise ValueError(f"{prop.name}: props are static. GamePiece is the kind that moves.")
        if not prop.scale > 0:
            raise ValueError(f"{prop.name}: scale must be above 0")
    for solid in world.solids:
        check_keypoints(solid)
    if len(world.of_type(Background)) > 1:
        raise ValueError("There can be only one Background")
    s = world.scenario
    if s.physics_hz % s.fps:
        raise ValueError(f"physics_hz ({s.physics_hz}) must be a multiple of fps ({s.fps})")
    for item in world.items:
        for f in dataclasses.fields(item):
            if isinstance(getattr(item, f.name), Range):
                raise AssertionError(f"{item.name}.{f.name} wasn't drawn")
