"""A school gym: a wood floor about the size of a basketball court, walls,
bleachers along one side, and ceiling lights."""

from sim import Box, Floor, Light, Range, Scene, Wall


COURT_SIZE = (30, 16)
ROW_DEPTH = 0.25
TOTAL_ROWS = 18
FLOOR_SIZE = (COURT_SIZE[0], COURT_SIZE[1] + ((TOTAL_ROWS * ROW_DEPTH) * 2))

class Gym(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=FLOOR_SIZE, texture=Range.choice(Floor.WOOD), roughness=Range(0.2, 0.6)))
        self.walls = self.add(Wall.around(self.floor, height=8, color=Range.color("white", "tan", "gray")))
        self.bleachers = self.generate_bleachers(rng)
        self.overhead = self.add(Light(
            kind="area", position=(0, 0, 7.9), size=(24, 12),
            intensity=Range(1500, 6000), color=Range.color("white", value=0.05),
        ))

    def generate_bleachers(self, rng):
        color = Range.color("gray", "blue", "red", "orange", "tan").draw(rng)
        result = []
        y_start = (FLOOR_SIZE[1] / 2) - (TOTAL_ROWS * ROW_DEPTH)
        for row in range(TOTAL_ROWS):
            height = (row + 1) * ROW_DEPTH
            y_pos = y_start + (row * ROW_DEPTH) + (ROW_DEPTH / 2)
            for y_mult in (1, -1):
                result.append(
                        self.add(Box(size=(COURT_SIZE[0], ROW_DEPTH, height),
                                     position=(0,  y_pos * y_mult, height / 2),
                                     color=color)))
        return result
