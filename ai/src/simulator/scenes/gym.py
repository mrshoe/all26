"""A school gym: a wood floor about the size of a basketball court, walls,
bleachers along one side, and ceiling lights."""

from sim import Box, Floor, Light, Range, Scene, Wall


class Gym(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(30, 18), texture=Range.choice(Floor.WOOD), roughness=Range(0.2, 0.6)))
        self.walls = self.add(Wall.around(self.floor, height=8, color=Range.color("white", "tan", "gray")))
        self.bleachers = self.add(Box(size=(20, 3, 2), position=(0, 7.5, 1), color=Range.color("gray", "blue")))
        self.overhead = self.add(Light(
            kind="area", position=(0, 0, 7.9), size=(24, 12),
            intensity=Range(1500, 6000), color=Range.color("white", value=0.05),
        ))
