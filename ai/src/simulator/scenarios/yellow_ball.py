"""A yellow ball tossed across the gym floor, seen by a camera at robot
height that follows it: training data for a ball detector."""

from sim import Camera, Range, Scenario, Sphere
from scenes.gym import Gym


class YellowBall(Scenario[Gym]):
    duration = 5.0  # seconds per episode
    fps = 10  # 50 frames per episode

    def setup(self, rng):
        self.scene.overhead.intensity = Range(500, 6000)  # sometimes dimmer than usual
        self.ball = self.add(Sphere(
            radius=0.09, color=Range.color("yellow"), label="ball", roughness=Range(0.3, 0.8),
            position=(rng.uniform(-2, 2), rng.uniform(1.5, 5), rng.uniform(0.1, 1.2)),
            velocity=(rng.uniform(-1.5, 1.5), rng.uniform(-1, 1), rng.uniform(0, 2)),
        ))
        self.front = self.add(Camera(
            position=(rng.uniform(-1, 1), rng.uniform(-1.5, -0.5), rng.uniform(0.3, 0.8)),
            # Close to the ball, but not always centered on it.
            look_at=self.ball if rng.random() < 0.7 else (rng.uniform(-2, 2), 3, 0.2),
            resolution=(640, 480), fov=Range(60, 75),
        ))
