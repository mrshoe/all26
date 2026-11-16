"""AprilTags on stands around the gym, from 0.5 to 10 m away and turned up
to 80 degrees, and a camera at robot height driving past them and turning,
with motion blur from a shutter drawn per episode: for finding how far,
how turned, and how fast a detector still reads tags. (Noise, brightness
and the like are training's augmentation, not the simulator's:
CAMERA-PLAN.md.)

Two cameras share each pose:
- `raw` is what the robot's AprilTag detector sees: the Pi Global Shutter
  camera's full 1408 x 1088 frame, about 940 px focal length (as a pinhole;
  the real lens's distortion isn't simulated);
- `standard` is the network's input, 640 x 480 at 80 degrees across
  (CAMERA-PLAN.md).
"""

import math

import numpy as np

from sim import AprilTag, Box, Camera, Range, Scenario
from scenes.gym import Gym

TAGS = 8
NEAR, FAR = 0.5, 10.0  # m from where the camera starts
SPREAD = 35  # degrees either side of where the camera first looks
MOST_TURNED = 80  # degrees off facing the camera
RAW = dict(resolution=(1408, 1088), fov=math.degrees(2 * math.atan(704 / 940)))
STANDARD = dict(resolution=(640, 480), fov=80)
BOARD = 0.02  # thickness of the board a tag is on
POLE = 0.04


class TagRange(Scenario[Gym]):
    duration = 2.0
    fps = 10

    def setup(self, rng):
        # Where the camera starts, and which way it looks at first (degrees from +x).
        self.start = np.array([rng.uniform(-6, 6), rng.uniform(-3, 3)])
        self.heading = rng.uniform(0, 360)
        self.height = rng.uniform(0.3, 0.9)
        # Robots drive sideways or backward past tags, turning; not through them.
        drive = math.radians(self.heading + rng.uniform(60, 300))
        speed = rng.uniform(0, 3.0)
        self.velocity = speed * np.array([math.cos(drive), math.sin(drive)])
        self.turning = rng.uniform(-90, 90)  # degrees a second
        self.tilt = rng.uniform(-5, 10)  # up

        ids = rng.choice(np.arange(1, 33), TAGS, replace=False)
        self.tags, self.boards, self.poles = [], [], []
        for tag in ids:
            self.place(int(tag), rng)

        # Both cameras get the same shutter: open 1 to 20 ms.
        shared = dict(position=self.at(0), rotation=self.turn(0), shutter=rng.uniform(0.001, 0.02))
        self.raw = self.add(Camera(**RAW, **shared))
        self.standard = self.add(Camera(**STANDARD, **shared))

    def place(self, tag: int, rng) -> None:
        """A tag on a board on a pole, somewhere in the camera's first view,
        turned from facing it."""
        distance = math.sqrt(rng.uniform(NEAR ** 2, FAR ** 2))  # as many per square meter near as far
        bearing = math.radians(self.heading + rng.uniform(-SPREAD, SPREAD))
        x, y = self.start + distance * np.array([math.cos(bearing), math.sin(bearing)])
        height = rng.uniform(0.3, 1.5)
        # Facing the camera (back along the bearing), then turned.
        yaw = math.degrees(bearing) + 180 + rng.uniform(-MOST_TURNED, MOST_TURNED)
        facing = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0)
        tag_ = AprilTag(tag=tag, position=(x, y, height), rotation=AprilTag.facing(facing))
        side = tag_.white() + 0.1
        # The board's front is where the tag's back is.
        board = np.array([x, y, height]) - np.array(facing) * BOARD / 2
        self.tags.append(self.add(tag_))
        self.boards.append(self.add(Box(size=(side, BOARD, side), position=tuple(board), rotation=(0, 0, yaw - 90),
                                        color=Range.color("white", "gray", "tan"))))
        top = height - side / 2
        if top > 0:
            pole = board - np.array(facing) * (BOARD / 2 + POLE / 2)
            self.poles.append(self.add(Box(size=(POLE, POLE, top), position=(pole[0], pole[1], top / 2),
                                           rotation=(0, 0, yaw - 90), color=Range.color("gray"))))

    def at(self, t: float) -> tuple:
        x, y = self.start + self.velocity * t
        return (float(x), float(y), self.height)

    def turn(self, t: float) -> tuple:
        # Cameras look along their -z; (90, 0, 0) looks along +y, so a
        # heading from +x is a turn of heading - 90 about z.
        return (90.0 + self.tilt, 0.0, self.heading + self.turning * t - 90.0)

    def step(self, t, rng):
        for cam in (self.raw, self.standard):
            cam.position = self.at(t)
            cam.rotation = self.turn(t)
