"""Scenes, Scenarios, naming and randomization. Run from ai/src/simulator:

    python -m unittest discover -s tests
"""

import unittest

import numpy as np

from sim import Box, Camera, Floor, Light, Range, Scenario, Scene, Sphere, Wall
from sim.world import build


class Room(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(10, 8), texture=Range.choice(Floor.WOOD)))
        self.walls = self.add(Wall.around(self.floor, height=3, color=Range.color("white")))
        self.lamp = self.add(Light(kind="area", position=(0, 0, 2.9), intensity=Range(100, 200)))
        self.add(Box(size=(1, 1, 1), position=(3, 3, 0.5)))  # kept in no attribute


class Toss(Scenario[Room]):
    def setup(self, rng):
        self.scene.lamp.intensity = Range(1000, 2000)
        self.ball = self.add(Sphere(radius=0.1, label="ball", position=(rng.uniform(-1, 1), 2, 1)))
        self.cam = self.add(Camera(position=(0, -2, 0.5), look_at=self.ball, fov=Range(60, 70)))


class TestBuild(unittest.TestCase):
    def test_names_come_from_attributes(self):
        world = build(Toss, seed=1)
        names = [item.name for item in world.items]
        self.assertEqual(names, ["floor", "walls.0", "walls.1", "walls.2", "walls.3", "lamp", "box.1", "ball", "cam"])
        self.assertEqual([item.id for item in world.items], list(range(1, 10)))

    def test_the_same_seed_draws_the_same_world(self):
        a, b, c = build(Toss, 7), build(Toss, 7), build(Toss, 8)
        self.assertEqual(a.drawn, b.drawn)
        self.assertNotEqual(a.drawn, c.drawn)

    def test_the_scenario_replaces_a_scene_range_before_it_is_drawn(self):
        for seed in range(20):
            self.assertGreaterEqual(build(Toss, seed).scene.lamp.intensity, 1000)

    def test_each_wall_draws_its_own_color(self):
        world = build(Toss, 3)
        colors = {item.color for item in world.scene.walls}
        self.assertEqual(len(colors), 4)
        for color in colors:
            self.assertTrue(all(0 <= c <= 1 for c in color))

    def test_ranges_dont_shift_what_setup_draws(self):
        class MoreRanges(Toss):
            def setup(self, rng):
                super().setup(rng)
                self.extra = self.add(Sphere(radius=Range(0.1, 0.2), position=(5, 0, 1)))

        self.assertEqual(build(Toss, 5).scenario.ball.position, build(MoreRanges, 5).scenario.ball.position)

    def test_a_scenario_can_remove_a_scene_object(self):
        class NoLamp(Toss):
            def setup(self, rng):
                super().setup(rng)
                self.remove(self.scene.lamp)

        world = build(NoLamp, 1)
        self.assertNotIn("lamp", [i.name for i in world.items])

    def test_a_name_used_by_both_is_an_error(self):
        class Clash(Scenario[Room]):
            def setup(self, rng):
                self.floor = self.add(Floor())

        with self.assertRaisesRegex(ValueError, "Both the Scene and the Scenario have an object called 'floor'"):
            build(Clash, 1)

    def test_a_scenario_must_say_its_scene(self):
        class NoScene(Scenario):
            pass

        with self.assertRaisesRegex(TypeError, r"Scenario\[MyScene\]"):
            build(NoScene, 1)

    def test_mistakes_are_caught_before_simulating(self):
        class BadTexture(Scenario[Room]):
            def setup(self, rng):
                self.scene.floor.texture = "lava"

        with self.assertRaisesRegex(ValueError, "texture must be one of"):
            build(BadTexture, 1)

    def test_settings_cant_change_once_started(self):
        world = build(Toss, 1)
        with self.assertRaisesRegex(AttributeError, "can't change once the episode has started"):
            world.scenario.ball.radius = 2
        world.scenario.ball.position = (0, 0, 3)
        self.assertEqual(world.pending[-1][1:], ("position", (0, 0, 3)))


class TestRange(unittest.TestCase):
    def test_draws(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            self.assertTrue(2 <= Range(2, 3).draw(rng) <= 3)
            self.assertIn(Range.choice("ab").draw(rng), "ab")
            self.assertAlmostEqual(float(np.linalg.norm(Range.sphere().draw(rng))), 1.0)
            r, g, b = Range.color("yellow").draw(rng)
            self.assertTrue(r > b and g > b)

    def test_unknown_color(self):
        with self.assertRaisesRegex(ValueError, "Unknown color 'yelow'"):
            Range.color("yelow")


if __name__ == "__main__":
    unittest.main()
