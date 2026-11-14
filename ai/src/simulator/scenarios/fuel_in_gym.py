"""A bunch of fuel on the gym floor"""

from sim import Camera, Range, Scenario, GamePiece
from scenes.gym import Gym


class YellowBall(Scenario[Gym]):
    duration = 2.0  # seconds per episode
    fps = 15

    def setup(self, rng):
        self.scene.overhead.intensity = Range(500, 6000)  # sometimes dimmer than usual
        self.fuel = [self.add(GamePiece(
            asset="frc/2026/fuel", label="fuel",
            position=(rng.uniform(-2, 2), rng.uniform(1.5, 5), rng.uniform(0.1, 1.2)),
            velocity=(rng.uniform(-1.5, 1.5), rng.uniform(-1, 1), rng.uniform(0, 2)),
        )) for _ in range(50)]
        self.front = self.add(Camera(
            position=(0, -5, 5),
            look_at=(0, 3, 3),
            resolution=(640, 480), fov=Range(60, 75),
        ))
