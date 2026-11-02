"""Groups: objects that go together, added to any Scene or Scenario. Run
from ai/src/simulator:

    python -m unittest discover -s tests
"""

import unittest
from dataclasses import dataclass

import numpy as np

from sim import Box, Camera, Floor, Group, Light, Range, Scenario, Scene, Sphere, Wall
from sim.runner import run
from sim.world import build


@dataclass(kw_only=True)
class Goal(Group):
    """Two posts and a crossbar, `width` apart, standing on the floor."""

    width: float = 2.0
    color: tuple = (1.0, 1.0, 1.0)

    def build(self, rng):
        self.posts = self.add([Box(size=(0.1, 0.1, 1), position=(x, 0, 0.5), color=self.color)
                               for x in (-self.width / 2, self.width / 2)])
        self.bar = self.add(Box(size=(self.width, 0.1, 0.1), position=(0, 0, 1.05), color=self.color))
        self.add(Sphere(radius=0.05, position=(0, 0, 1.2), static=True))  # kept in no attribute


@dataclass(kw_only=True)
class Pitch(Group):
    """A Group of Groups: a goal at each end."""

    length: float = 10.0

    def build(self, rng):
        self.goals = self.add([Goal(position=(0, y, 0), rotation=(0, 0, turn))
                               for y, turn in ((-self.length / 2, 0), (self.length / 2, 180))])
        self.lamp = self.add(Light(kind="spot", position=(0, 0, 3), direction=(1, 0, 0)))


class Empty(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(30, 30)))


class OneGoal(Scenario[Empty]):
    duration = 0.5

    def setup(self, rng):
        self.goal = self.add(Goal(position=(1, 2, 0), rotation=(0, 0, 90)))


def by_name(world):
    return {item.name: item for item in world.items}


class TestGroups(unittest.TestCase):
    def test_objects_are_named_under_the_group(self):
        names = list(by_name(build(OneGoal, 1)))
        self.assertEqual(names, ["floor", "goal.posts.0", "goal.posts.1", "goal.bar", "goal.sphere.1"])

    def test_objects_are_placed_relative_to_the_group(self):
        items = by_name(build(OneGoal, 1))
        # Turned 90 degrees about z, then moved to (1, 2, 0): x becomes y.
        np.testing.assert_allclose(items["goal.posts.0"].position, (1, 1, 0.5), atol=1e-9)
        np.testing.assert_allclose(items["goal.posts.1"].position, (1, 3, 0.5), atol=1e-9)
        np.testing.assert_allclose(items["goal.bar"].rotation, (0, 0, 90), atol=1e-9)

    def test_groups_nest(self):
        class Match(Scenario[Empty]):
            def setup(self, rng):
                self.pitch = self.add(Pitch(position=(0, 0, 0), rotation=(0, 0, 90)))

        world = build(Match, 1)
        items = by_name(world)
        self.assertIn("pitch.goals.1.bar", items)
        # The far goal: at y = +5 in the pitch, turned 180; the pitch is turned 90.
        np.testing.assert_allclose(items["pitch.goals.1.posts.0"].position, (-5, 1, 0.5), atol=1e-9)
        np.testing.assert_allclose(np.cos(np.radians(items["pitch.goals.1.bar"].rotation[2])), np.cos(np.radians(270)),
                                   atol=1e-9)
        # A light's direction turns with its Group.
        np.testing.assert_allclose(items["pitch.lamp"].direction, (0, 1, 0), atol=1e-9)
        self.assertEqual(set(world.groups), {"pitch", "pitch.goals.0", "pitch.goals.1"})

    def test_settings_are_drawn_before_build_and_recorded(self):
        class Wide(Scenario[Empty]):
            def setup(self, rng):
                self.goal = self.add(Goal(width=Range(3, 4), position=(Range(-1, 1), 0, 0)))

        world = build(Wide, 2)
        width = world.groups["goal"]["width"]
        self.assertTrue(3 <= width <= 4)
        self.assertAlmostEqual(by_name(world)["goal.bar"].size[0], width)
        self.assertEqual(world.groups["goal"]["type"], "Goal")
        self.assertEqual(build(Wide, 2).groups, world.groups)  # same seed, same draws

    def test_a_scenario_can_move_or_remove_a_scene_group(self):
        class GoalScene(Scene):
            def setup(self, rng):
                self.floor = self.add(Floor())
                self.goal = self.add(Goal())
                self.other = self.add(Goal(position=(5, 0, 0)))

        class Moved(Scenario[GoalScene]):
            def setup(self, rng):
                self.scene.goal.position = (0, Range(3, 4), 0)
                self.remove(self.scene.other)

        world = build(Moved, 1)
        items = by_name(world)
        self.assertTrue(3 <= items["goal.bar"].position[1] <= 4)
        self.assertFalse(any(name.startswith("other.") for name in items))
        self.assertNotIn("other", world.groups)

    def test_an_unnamed_group_gets_a_numbered_name(self):
        class Unnamed(Scenario[Empty]):
            def setup(self, rng):
                self.add([Goal(), Goal(position=(3, 0, 0))])

        self.assertIn("goal.2.bar", by_name(build(Unnamed, 1)))

    def test_walls_and_cameras_move_with_the_group(self):
        @dataclass(kw_only=True)
        class Booth(Group):
            def build(self, rng):
                self.wall = self.add(Wall(start=(0, 0), end=(2, 0)))
                self.cam = self.add(Camera(position=(1, -1, 1), look_at=(1, 0, 1)))

        class WithBooth(Scenario[Empty]):
            def setup(self, rng):
                self.booth = self.add(Booth(position=(10, 0, 0), rotation=(0, 0, 90)))

        items = by_name(build(WithBooth, 1))
        np.testing.assert_allclose(items["booth.wall"].end, (10, 2), atol=1e-9)
        np.testing.assert_allclose(items["booth.cam"].look_at, (10, 1, 1), atol=1e-9)

        class Tilted(Scenario[Empty]):
            def setup(self, rng):
                self.booth = self.add(Booth(rotation=(10, 0, 0)))

        with self.assertRaisesRegex(ValueError, "can only turn about z"):
            build(Tilted, 1)

    def test_mistakes(self):
        @dataclass(kw_only=True)
        class WithFloor(Group):
            def build(self, rng):
                self.add(Floor())

        class Bad(Scenario[Empty]):
            def setup(self, rng):
                self.add(WithFloor())

        with self.assertRaisesRegex(TypeError, "A Floor can't be in a Group"):
            build(Bad, 1)

        class Twice(Scenario[Empty]):
            def setup(self, rng):
                goal = self.add(Goal())
                self.add(goal)

        with self.assertRaisesRegex(ValueError, "already added"):
            build(Twice, 1)

        with self.assertRaisesRegex(RuntimeError, "only be added in setup"):
            Empty().add(Goal())

    def test_physics_sees_the_placed_objects(self):
        class Drop(Scenario[Empty]):
            duration = 1.0

            def setup(self, rng):
                self.goal = self.add(Goal(position=(1, 2, 0), rotation=(0, 0, 90)))
                # Turned 90 degrees, the crossbar runs along y from (1, 1) to (1, 3)
                # at 1.0-1.1 m: drop the ball onto it, away from the little sphere.
                self.ball = self.add(Sphere(radius=0.05, position=(1, 2.5, 1.6), label="ball"))

        world = run(Drop, 1, "physics").world
        self.assertGreater(world.scenario.ball.position[2], 1.0)  # resting on the bar, not the floor


if __name__ == "__main__":
    unittest.main()
