"""Crates tossed onto the gym floor, with their 8 corners as keypoints, and
balls rolling in front of them: an example of keypoints, and of full boxes
around things partly hidden."""

from sim import Box, Camera, Range, Scenario, Sphere
from scenes.gym import Gym


class CrateCorners(Scenario[Gym]):
    duration = 2.0
    fps = 10

    def setup(self, rng):
        self.crates = []
        for i in range(4):
            size = tuple(rng.uniform(0.3, 0.7, 3))
            self.crates.append(self.add(Box(
                size=size, label="crate", static=False, mass=5, color=Range.color("tan", "orange", "gray"),
                keypoints={"corner": Box.corners(size)},
                position=(-1.5 + i, rng.uniform(3, 4.5), rng.uniform(0.5, 1.5)),
                rotation=(Range(0, 360), Range(0, 360), Range(0, 360)),
                velocity=(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0),
            )))
        self.balls = [self.add(Sphere(
            radius=0.12, label="ball", color=Range.color("yellow"),
            position=(rng.uniform(-2, 2), rng.uniform(1.5, 2.5), 0.12),
            velocity=(rng.uniform(-1.5, 1.5), 0, 0),
        )) for _ in range(3)]
        self.front = self.add(Camera(position=(0, -0.5, 0.6), look_at=(0, 3, 0.3), resolution=(640, 480),
                                     fov=Range(60, 75)))
