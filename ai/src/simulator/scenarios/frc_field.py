"""The 2026 field in the gym, and a camera at robot height driving across the
neutral zone, looking toward the blue HUB: FUEL, HUBs, BUMPs and AprilTags
from a robot's view. A few extra FUEL are tossed in to bounce around."""

from sim import Camera, GamePiece, Range, Scenario
from sim.frc import FrcField
from scenes.gym import Gym

START, END = -3.2, 3.2  # the camera's y, from one side of the field to the other


class FrcFieldDrive(Scenario[Gym]):
    duration = 4.0
    fps = 10

    def setup(self, rng):
        self.scene.overhead.intensity = Range(3000, 8000)
        self.field = self.add(FrcField(year=2026, rotation=(0, 0, Range(-1, 1))))
        self.tossed = [self.add(GamePiece(
            asset="frc/2026/fuel", label="fuel",
            position=(rng.uniform(-1.5, 1.5), rng.uniform(-3, 3), rng.uniform(0.8, 2.0)),
            velocity=(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 2)),
        )) for _ in range(12)]
        self.x = rng.uniform(-2.4, -1.8)
        self.height = rng.uniform(0.4, 0.8)
        self.cam = self.add(Camera(position=(self.x, START, self.height), look_at=(3.5, 0, 0.8),
                                   resolution=(640, 480), fov=Range(70, 85)))

    def step(self, t, rng):
        y = START + (END - START) * t / self.duration
        self.cam.position = (self.x, y, self.height)
        self.cam.look_at = (3.5, y * 0.4, 0.8)
