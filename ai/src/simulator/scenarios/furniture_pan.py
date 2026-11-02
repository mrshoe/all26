"""A row of furniture in front of the gym's bleachers, and a camera at robot
height driving past it in a straight line: training data for furniture,
and a check that props look right from every side."""

from sim import Camera, Prop, Range, Scenario, Sphere, assets
from scenes.gym import Gym

LABELS = ("chair", "table", "couch", "bench", "cabinet", "shelf", "desk", "stool", "plant")
START, END = -9.0, 9.0  # the camera's x, from one end to the other


class FurniturePan(Scenario[Gym]):
    duration = 6.0
    fps = 10

    def setup(self, rng):
        # A labeled row along y = 5, in front of the bleachers.
        models = [(label, m) for label in LABELS for m in assets.find(label)]
        self.furniture = []
        x = START - 1
        while x < END + 1:
            label, model = models[int(rng.integers(len(models)))]
            width = max(assets.info(model)["size"][:2])
            x += width / 2
            self.furniture.append(self.add(Prop(
                asset=model, label=label, position=(x, rng.uniform(4.4, 5.0), 0),
                rotation=(0, 0, Range(150, 210)),  # roughly facing the court
                tint=Range.color("white", "tan", saturation=0.1, value=0.1),
            )))
            x += width / 2 + rng.uniform(0.3, 1.2)

        self.balls = [self.add(Sphere(
            radius=0.15, color=Range.color("yellow"), label="ball", roughness=Range(0.3, 0.8),
            position=(rng.uniform(-8, 8), rng.uniform(1.5, 5), rng.uniform(0.5, 3.5)),
            velocity=(rng.uniform(-1.5, 1.5), rng.uniform(-1, 1), rng.uniform(0, 2)),
        )) for _ in range(100)]

        self.height = rng.uniform(0.8, 1.2)
        self.y = rng.uniform(-3.0, 0.0)
        self.front = self.add(Camera(
            position=(START, self.y, self.height), look_at=(START, self.y + 1.0, self.height),
            resolution=(640, 480), fov=Range(65, 75),
        ))

    def step(self, t, rng):
        x = START + (END - START) * t / self.duration
        self.front.position = (x, self.y, self.height)
        self.front.look_at = (x, 5, 0.4)
