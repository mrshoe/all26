# The simulator

Generates labeled pictures in simulation. You write **Scenes** (places, like
a gym) and **Scenarios** (what happens there); the `sim` framework does the
rest: physics, rendering, labels, logging, preview videos and run records.

```
sim/          the framework (don't need to touch it to write scenarios)
scenes/       environments: gym.py, ...
scenarios/    what happens in them: yellow_ball.py, ...
tests/        python -m unittest discover -s tests
```

## Writing a Scene

A Scene is somewhere things happen: floor, walls, lights, static objects.
Keep each object in an attribute; the attribute's name becomes the object's
name in the logs and the visualizer.

```python
from sim import Box, Floor, Light, Range, Scene, Wall

class Gym(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(30, 18), texture=Range.choice(Floor.WOOD)))
        self.walls = self.add(Wall.around(self.floor, height=8, color=Range.color("white", "tan")))
        self.bleachers = self.add(Box(size=(20, 3, 2), position=(0, 7.5, 1)))
        self.overhead = self.add(Light(kind="area", position=(0, 0, 7.9), size=(24, 12),
                                       intensity=Range(1500, 6000)))
```

## Writing a Scenario

A Scenario says which Scene it's in, `Scenario[Gym]`, then changes the
Scene's objects through `self.scene`, adds its own, and can move things every
physics step in `step`.

```python
from sim import Camera, Range, Scenario, Sphere
from scenes.gym import Gym

class YellowBall(Scenario[Gym]):
    duration = 5.0  # seconds per episode
    fps = 10        # frames per second, so 50 frames

    def setup(self, rng):
        self.scene.overhead.intensity = Range(500, 6000)
        self.ball = self.add(Sphere(radius=0.09, color=Range.color("yellow"), label="ball",
                                    position=(rng.uniform(-2, 2), 3, 1)))
        self.front = self.add(Camera(position=(0, 0, 0.5), look_at=self.ball, fov=Range(60, 75)))

    def step(self, t, rng):
        if t > 2 and self.ball.position[2] < 0.1:
            self.ball.push((0, 0, 0.5))  # kick it back up
```

- **Units** are meters, kilograms and seconds; angles are in degrees. z is up.
- **Randomness**: every `Range(...)`, and everything drawn from `rng`, comes
  from the episode's seed, so `--seed` replays an episode exactly. `Range`s
  are drawn after both `setup`s, so a Scenario can replace a Scene's Range.
- **Labels**: give the things a detector should find a `label`. The
  framework records each labeled object's exact box in every camera, and
  its full box (below).
- **Keypoints**: named points on an object, for keypoint detectors (below).
- **What's available**: `Sphere`, `Box`, `Floor` (textures in
  `Floor.TEXTURES`), `Wall` (and `Wall.around(floor)`), `Light` (`sun`,
  `area`, `point`, `spot`), `Camera`, `Background`, `Prop`, `GamePiece`,
  `Group` and `FrcField` (below). See `sim/objects.py`.
- **Moving things in `step`**: set `position`, `rotation` or `velocity`, or
  call `push(impulse)`. Static things (`static=True`, the default for boxes,
  floors and walls) can't move.

## Keypoints and full boxes

Give any object named points, in its own coordinates (before its
`rotation` and `position`), and the framework records where each lands in
every camera's picture, every frame, and whether it's seen:

```python
size = (0.5, 0.4, 0.3)
self.crate = self.add(Box(size=size, label="crate", keypoints={"corner": Box.corners(size)}))
self.goal = self.add(Box(size=(2, 0.1, 1), keypoints={"post_top": [(-1, 0, 0.5), (1, 0, 0.5)]}))
```

- **Every point is recorded**, seen or hidden, so whoever trains chooses
  which to use. Each says whether it's in the picture, whether it's seen,
  and if not, what hides it: the object itself (`"self"`, its far side) or
  the thing in front. A point behind the camera has no pixel, so it isn't
  recorded. A Prop's own coordinates have z = 0 at its base.
- **Seen** comes from a depth image Blender renders next to the mask: a
  point is hidden when what's drawn at its pixel, and at every pixel next
  to it, is more than a few centimeters in front of it.
- **Full boxes**: each labeled object's box is around the part that's seen.
  Its full box is around its whole shape, as if nothing hid it (a Prop's
  from its collision hulls, so a little loose around chair legs), cut to
  the picture like the other. With it comes how many pixels the whole
  object would cover, so `box.share` says how much of it is seen.
- **Reading them**: `ep.keypoints("front", 12)` and `ep.boxes("front", 12)`
  (`box.full`, `box.share`), or `sample.keypoints()` from a dataset. For
  YOLO, `ds.yolo(classes, boxes="full", min_share=0.25)` exports full boxes
  of objects at least a quarter seen.
- **The visualizer** draws them over the video: dashed full boxes around
  partly hidden objects, and a dot for each keypoint, filled if it's seen
  and hollow if it's hidden. Chips under the video turn each on and off.

`scenarios/crate_corners.py` is an example.

## Groups: objects that go together

A Group is a set of objects written once and added to any Scene or
Scenario, e.g. an FRC field that goes in the gym for one Scenario and a
parking lot for another. Subclass `Group`, give it settings, and add its
objects in `build`, placed relative to the Group:

```python
from dataclasses import dataclass
from sim import Box, Group, Range

@dataclass(kw_only=True)
class Goal(Group):
    width: float = 2.0

    def build(self, rng):
        self.posts = self.add([Box(size=(0.1, 0.1, 1), position=(x, 0, 0.5))
                               for x in (-self.width / 2, self.width / 2)])
        self.bar = self.add(Box(size=(self.width, 0.1, 0.1), position=(0, 0, 1.05)))

# In any Scene's or Scenario's setup:
self.goal = self.add(Goal(width=Range(2, 3), position=(4, 0, 0), rotation=(0, 0, Range(0, 360))))
```

- **Placing**: its objects' positions and rotations (and velocities, light
  directions, cameras' look-at points and walls) are relative to the
  Group's `position` and `rotation`. Once the episode starts, they're in the
  world's coordinates like everything else.
- **Names**: its objects are named under it: `goal.posts.0`, `goal.bar`.
  Groups can hold Groups (`pitch.goals.1.bar`).
- **Settings** are drawn when it's added, so `build` sees numbers, not
  Ranges. They're recorded in episode.json under `groups`, and `sim check`
  prints them. Its `position` and `rotation` can still be changed until the
  episode starts, so a Scenario can move a Scene's Group, or `remove()` it.
- **Not in a Group**: floors and backgrounds, which belong to a Scene. A
  Group with walls can only turn about z.

## Props: furniture and other models

`Prop` puts a model from the asset catalog in the world: about 300 chairs,
tables, benches, plants, crates, tools and so on from
[Poly Haven](https://polyhaven.com/models) (CC0), each cut down to at most
20,000 triangles. `position` is the middle of its base, so z = 0 stands it
on the floor.

```python
from sim import Prop, Range, assets

assets.labels()                         # {"chair": 13, "table": 22, ...}
assets.find("chair")                    # every chair's id
assets.find(category="furniture/seating", max_size=1.0)
assets.info("ArmChair_01")              # its size in meters, category, ...

self.chairs = [self.add(Prop(
    asset=Range.choice(assets.find("chair")), label="chair",
    position=(rng.uniform(-3, 3), rng.uniform(2, 6), 0), rotation=(0, 0, Range(0, 360)),
    scale=Range(0.9, 1.1), tint=Range.color("white", "tan"),
)) for _ in range(4)]
```

- **Labels**: a Prop has no label unless you give it one, like any object.
  The catalog's labels are only for finding models.
- **tint** multiplies the model's own colors; (1, 1, 1) leaves them.
- **Static.** Physics uses each model's convex parts, so a ball can roll
  under a table. `GamePiece` is the kind of Prop that moves (below).
- **On your laptop**, `check` doesn't have the models, so physics uses each
  Prop's box from the catalog instead (and says so). Cloud jobs download
  the models they use.

`sim/assets/polyhaven.json` is the catalog. Admins make it, and the models in
the bucket, with `python -m sim.assets.polyhaven` (see its docstring).

## The FRC field

`FrcField` is FIRST's official field for a season, from FIRST's own CAD, as a
Group, so it goes in any Scene or Scenario:

```python
from sim import Range
from sim.frc import FrcField

self.field = self.add(FrcField(year=2026, rotation=(0, 0, Range(-2, 2))))
self.field.hub              # [hub_blue, hub_red], labeled "hub"
self.field.fuel             # the game pieces at their starting spots
self.remove(self.field.outpost[0])
self.field.to_wpilib(self.cam.position, yaw=90)   # where robot code would say it is
```

- **Its parts**: each element (HUB, TOWER, BUMP, TRENCH, OUTPOST, DEPOT,
  the perimeter, the carpet and its tape lines, the scoring table) is a
  static Prop named after what it is: `field.hub.0`. Elements the season's
  config labels (e.g. the HUB) get boxes. AprilTags have their real
  images, so tag detection can be trained and tested.
- **Game pieces** are `GamePiece`s: Props that physics moves, with mass,
  bounciness and friction from the catalog. `starting_pieces=False` leaves
  them out, and you can add your own:
  `GamePiece(asset="frc/2026/fuel", label="fuel", position=(1, 2, 0.5))`.
- **Choosing parts**: `elements=("hub", "carpet")` keeps only those.
- **Placing it**: its `position` is the field's center, with the carpet at
  z = 0, so a Scene's floor stays under it.
- **Models** are `frc/<year>/<name>` in the asset catalog
  (`assets.info("frc/2026/hub_red")`), from `sim/assets/frc/<year>.json`.
  An admin imports a new season on kickoff day with
  `python -m sim.assets.frc <year>`; see `sim/assets/frc/importer.py`.

## Running it

Set up once, with Python 3.11 ([uv](https://docs.astral.sh/uv/)):

```sh
cd ai/src/simulator
uv venv --python 3.11 && uv pip install -r requirements.txt
```

Check a Scenario on your laptop. This runs the physics without rendering,
prints what was drawn, and says how often each camera sees each labeled
object:

```sh
.venv/bin/python -m sim check scenarios/yellow_ball.py --episodes 5
```

Render one episode on your laptop (a few seconds a frame) into
`out/yellow_ball/seed-0/`:

```sh
.venv/bin/python -m sim run scenarios/yellow_ball.py --seed 0
```

Run many episodes in the cloud, then see them in the visualizer:

```sh
wildhat launch --episodes 20 --timeout 2h -- python -m sim run scenarios/yellow_ball.py
```

## What an episode writes

| File | What |
|---|---|
| `episode.json` | Scenario, Scene, backend, seed, timing, and which tables there are |
| `objects.parquet` | Every object: id (the one in the masks), name, type, label, static, and what was drawn |
| `cameras.parquet` | Each camera's resolution and intrinsics |
| `sim.parquet` | Per frame: time, and how long `step()`, physics, rendering and logging took |
| `poses.parquet` | Per frame, each moving object, camera and light: position, rotation, linear and angular velocity (static things once, at frame 0) |
| `images.parquet` | Per frame, each camera's picture and mask |
| `boxes.parquet` | One row per labeled object seen by a camera: its box in pixels and how many pixels it covers, and its full box and how many it would cover if nothing hid it |
| `keypoints.parquet` | One row per keypoint per camera per frame: pixel, depth, in the picture or not, seen or not, and what hides it |
| `cam/<camera>/000000.jpg` | What each camera saw, per frame |
| `mask/<camera>/000000.png` | Which object each pixel shows (16-bit object ids, from `objects`) |
| `preview/<camera>.mp4` | A small preview video for the visualizer, which draws the labels over it from the tables |

Tables join on `frame` and `object` (a camera is an object too). The
schemas are in `ai/src/shared/logger.py`: the shared `Logger` and
`PostProcessor` write these, as they will on the robot, and
`sim/logger.py` turns the simulation into their rows.
`ai/src/shared/episodes.py` reads them back.

## Backends

Scenes and Scenarios never touch the renderer, so it can change without
anyone's code changing. Today, `blender` uses MuJoCo for physics and Blender's
Cycles (on the CPU) for pictures; `physics` is MuJoCo alone, for `check`.
Isaac Sim will be another backend once the project has GPUs.
