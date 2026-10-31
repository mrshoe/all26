"""The Blender backend renders what physics says, with exact masks. Skipped
without bpy (pip install bpy==4.5.4, Python 3.11)."""

import unittest

import numpy as np

from sim import Box, Camera, Floor, Light, Scenario, Scene, Sphere
from sim.geometry import intrinsics, project
from sim.runner import camera_pose, read_poses
from sim.world import build

try:
    from sim.backends.blender import Blender
except ImportError:
    Blender = None


class Lit(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(10, 10), texture="wood"))
        self.sun = self.add(Light(kind="sun", intensity=4, direction=(0.3, 0.5, -1)))


class Still(Scenario[Lit]):
    def setup(self, rng):
        self.ball = self.add(Sphere(radius=0.2, label="ball", color=(1, 0.85, 0), position=(0.4, 3, 0.2)))
        self.crate = self.add(Box(size=(0.5, 0.5, 0.5), position=(-1, 4, 0.25), color=(0.2, 0.3, 0.8)))
        self.cam = self.add(Camera(position=(0, 0, 0.6), look_at=(0, 3, 0.2), resolution=(320, 240), fov=60))


@unittest.skipIf(Blender is None, "bpy isn't installed")
class TestBlender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = build(Still, 0)
        backend = Blender(samples=4)
        backend.build(cls.world)
        read_poses(cls.world, backend)
        cam = cls.world.cameras[0]
        cls.pose = camera_pose(cam)
        cls.picture = backend.render(cam, *cls.pose)
        backend.close()

    def test_sizes(self):
        self.assertEqual(self.picture.rgb.shape, (240, 320, 3))
        self.assertEqual(self.picture.ids.shape, (240, 320))

    def test_the_mask_holds_exactly_the_objects_ids(self):
        ids = set(np.unique(self.picture.ids).tolist())
        floor, ball, crate = (self.world.by_name(n).id for n in ("floor", "ball", "crate"))
        self.assertEqual(ids, {0, floor, ball, crate})

    def test_the_ball_is_where_the_camera_math_says(self):
        ball = self.world.by_name("ball")
        k = intrinsics((320, 240), 60)
        cx, cy = project(ball.position, *self.pose, k)
        ys, xs = np.nonzero(self.picture.ids == ball.id)
        self.assertAlmostEqual(xs.mean(), cx, delta=1.5)
        self.assertAlmostEqual(ys.mean(), cy, delta=1.5)
        # A sphere's outline is about as wide as fx * 2r / distance.
        distance = np.linalg.norm(np.subtract(ball.position, self.pose[0]))
        self.assertAlmostEqual(xs.max() - xs.min() + 1, k["fx"] * 0.4 / distance, delta=3)

    def test_the_ball_is_yellow(self):
        ball = self.world.by_name("ball")
        r, g, b = self.picture.rgb[self.picture.ids == ball.id].mean(0)
        self.assertGreater(r, 2 * b)
        self.assertGreater(g, 2 * b)


if __name__ == "__main__":
    unittest.main()
