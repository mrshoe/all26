"""The 2025 REEFSCAPE field in the gym, and a camera at robot height driving
toward one of the REEFs from a random side, as a robot lining up to score
would: REEF BRANCHes and troughs as keypoints, ALGAE and CORAL, AprilTags."""

import math

from sim import Camera, Range, Scenario
from sim.frc import FrcField
from scenes.gym import Gym

# The REEFs' centers on the 2025 field (x, y), from the catalog's layout.
REEFS = [(-4.285, 0.0), (4.285, 0.0)]
REEF_MIDDLE = 0.9  # m up: about halfway between L1 and L4


class ReefApproach(Scenario[Gym]):
    duration = 3.0
    fps = 10

    def setup(self, rng):
        self.scene.overhead.intensity = Range(3000, 8000)
        self.field = self.add(FrcField(year=2025))
        cx, cy = REEFS[rng.integers(len(REEFS))]
        self.reef = (cx, cy, REEF_MIDDLE)
        # Toward one of its six faces, give or take, from 3.5 m to 1.3 m away.
        heading = math.radians(60 * rng.integers(6) + rng.uniform(-20, 20))
        self.toward = (math.cos(heading), math.sin(heading))
        self.far, self.near = rng.uniform(3.0, 3.8), rng.uniform(1.2, 1.5)
        self.height = rng.uniform(0.3, 0.9)
        self.cam = self.add(Camera(position=self.at(0.0), look_at=self.reef, resolution=(640, 480),
                                   fov=Range(70, 90)))

    def at(self, t: float) -> tuple:
        distance = self.far + (self.near - self.far) * t / self.duration
        return (self.reef[0] + self.toward[0] * distance, self.reef[1] + self.toward[1] * distance, self.height)

    def step(self, t, rng):
        self.cam.position = self.at(t)
