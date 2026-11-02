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
  framework records each labeled object's exact box in every camera.
- **What's available**: `Sphere`, `Box`, `Floor` (textures in
  `Floor.TEXTURES`), `Wall` (and `Wall.around(floor)`), `Light` (`sun`,
  `area`, `point`, `spot`), `Camera`, `Background`, and `Prop` (below).
  See `sim/objects.py`.
- **Moving things in `step`**: set `position`, `rotation` or `velocity`, or
  call `push(impulse)`. Static things (`static=True`, the default for boxes,
  floors and walls) can't move.

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
- **Static only, for now.** Physics uses each model's convex parts, so a
  ball can roll under a table.
- **On your laptop**, `check` doesn't have the models, so physics uses each
  Prop's box from the catalog instead (and says so). Cloud jobs download
  the models they use.

`sim/assets/polyhaven.json` is the catalog. Admins make it, and the models in
the bucket, with `python -m sim.assets.polyhaven` (see its docstring).

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
| `boxes.parquet` | One row per labeled object seen by a camera: its box in pixels and how many pixels it covers |
| `cam/<camera>/000000.jpg` | What each camera saw, per frame |
| `mask/<camera>/000000.png` | Which object each pixel shows (16-bit object ids, from `objects`) |
| `preview/<camera>.mp4`, `.boxes.mp4` | Small preview videos for the visualizer, without and with boxes |

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
