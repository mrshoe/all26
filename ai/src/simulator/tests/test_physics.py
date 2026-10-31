"""Geometry, MuJoCo physics and the episode loop, with no rendering."""

import math
import unittest

import numpy as np

from sim import Box, Camera, Floor, Scenario, Scene, Sphere
from sim import runner
from sim.geometry import euler_from_quat, intrinsics, project, quat_from_euler, quat_looking_at


class TestGeometry(unittest.TestCase):
    def test_euler_round_trip(self):
        for angles in [(0, 0, 0), (10, 20, 30), (-45, 60, 170), (90, 0, 0)]:
            np.testing.assert_allclose(euler_from_quat(quat_from_euler(angles)), angles, atol=1e-6)

    def test_a_camera_looking_at_a_point_sees_it_in_the_middle(self):
        eye, target = (1, -3, 0.5), (0, 2, 0.2)
        k = intrinsics((640, 480), 70)
        x, y = project(target, eye, quat_looking_at(eye, target), k)
        self.assertAlmostEqual(x, 320, places=6)
        self.assertAlmostEqual(y, 240, places=6)
        # Up in the world is up in the picture.
        x, y = project((0, 2, 1.2), eye, quat_looking_at(eye, target), k)
        self.assertLess(y, 240)
        self.assertIsNone(project((1, -5, 0.5), eye, quat_looking_at(eye, target), k))


class Empty(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(20, 20)))


def drop(bounciness, height=1.0, **kwargs):
    class Drop(Scenario[Empty]):
        duration, fps = 3.0, 60

        def setup(self, rng):
            self.ball = self.add(Sphere(radius=0.1, position=(0, 0, height), bounciness=bounciness, **kwargs))
            self.cam = self.add(Camera(position=(0, -3, 0.5), look_at=self.ball))

    return Drop


class Heights:
    """A logger that keeps the ball's height each frame."""

    def __init__(self, world):
        self.world, self.z = world, []

    def frame(self, index, t, cameras, pictures):
        self.z.append(self.world.scenario.ball.position[2])


def heights(scenario, seed=0):
    out = {}
    runner.run(scenario, seed, "physics", lambda world: out.setdefault("log", Heights(world)))
    return np.array(out["log"].z)


def bounce_peaks(z):
    return [z[i] for i in range(1, len(z) - 1) if z[i - 1] < z[i] >= z[i + 1] and z[i] > 0.11]


class TestPhysics(unittest.TestCase):
    def test_a_ball_falls_and_lands_on_the_floor(self):
        z = heights(drop(0.0))
        self.assertAlmostEqual(z[0], 1.0, places=6)
        self.assertAlmostEqual(z[-1], 0.1, delta=0.005)
        # Free fall: 1 - g t²/2 after a quarter second.
        self.assertAlmostEqual(z[15], 1 - 9.81 * 0.25**2 / 2, delta=0.01)

    def test_bounciness_is_the_coefficient_of_restitution(self):
        for e in (0.3, 0.6, 0.9):
            first = bounce_peaks(heights(drop(e)))[0]
            # It bounces back e² as high, measured from where it touches.
            self.assertAlmostEqual((first - 0.1) / 0.9, e * e, delta=0.08, msg=f"bounciness {e}")

    def test_very_bouncy_balls_stay_put(self):
        z = heights(drop(1.0))
        self.assertLess(z.max(), 1.0001)

    def test_the_same_seed_gives_the_same_motion(self):
        class Rolling(drop(0.6)):
            def setup(self, rng):
                super().setup(rng)
                self.ball.velocity = (rng.uniform(-1, 1), rng.uniform(-1, 1), 0)

        np.testing.assert_array_equal(heights(Rolling, 3), heights(Rolling, 3))

    def test_step_can_move_and_push_things(self):
        class Kick(drop(0.0)):
            def step(self, t, rng):
                if math.isclose(t, 1.0, abs_tol=1e-9):
                    self.ball.position = (0, 0, 2)
                if math.isclose(t, 2.0, abs_tol=1e-9):
                    self.ball.push((0, 0, 0.2 * 3))  # 3 m/s up

        z = heights(Kick)
        self.assertGreater(z[61], 1.9)  # teleported at 1s
        self.assertGreater(z[125], 0.3)  # kicked up at 2s

    def test_rotation_is_read_from_physics(self):
        seen = []

        class Spin(drop(0.0)):
            def setup(self, rng):
                super().setup(rng)
                self.ball.velocity = (1.0, 0, 0)  # rolls, so it turns

            def step(self, t, rng):
                if math.isclose(t, 2.0, abs_tol=1e-9):
                    seen.append((self.ball.rotation, self.ball.quat.copy()))

        heights(Spin)
        (rotation, quat), = seen
        np.testing.assert_allclose(quat_from_euler(rotation), quat, atol=1e-9)
        self.assertGreater(max(abs(a) for a in rotation), 1)

    def test_static_things_cant_move(self):
        class MoveWall(drop(0.0)):
            def setup(self, rng):
                super().setup(rng)
                self.crate = self.add(Box(position=(2, 0, 0.5)))

            def step(self, t, rng):
                self.crate.position = (3, 0, 0.5)

        with self.assertRaisesRegex(RuntimeError, "crate is static"):
            heights(MoveWall)

    def test_check_counts_frames_in_view(self):
        episode = runner.run(drop(0.5), 0, "physics")
        self.assertEqual(episode.in_view, {"cam": {}})
        labeled = drop(0.5, label="ball")
        episode = runner.run(labeled, 0, "physics")
        self.assertEqual(episode.in_view["cam"]["ball"], episode.frames)


if __name__ == "__main__":
    unittest.main()
