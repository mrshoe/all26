"""Keypoints and full boxes from geometry (sim/labels.py), without Blender:
pictures here are made up. test_blender checks them against real renders."""

import unittest

import numpy as np

from sim import AprilTag, Box, Camera, Floor, Range, Scenario, Scene, Sphere
from sim import labels
from sim.geometry import intrinsics, project, quat_from_euler, quat_looking_at
from sim.world import build

K = intrinsics((320, 240), 90)
# At the origin, looking along +y (the world's), with +z up in the picture.
LOOKING_NORTH = quat_from_euler((90, 0, 0))


def view(position=(0, 0, 0), quat=LOOKING_NORTH):
    return labels.View(position, quat, K)


class Empty(Scene):
    pass


class Crates(Scenario[Empty]):
    def setup(self, rng):
        size = (0.4, 0.6, 0.2)
        self.crate = self.add(Box(size=size, position=(0.3, 4, 0.1), rotation=(0, 0, 30),
                                  keypoints={"corner": Box.corners(size), "top": [(0, 0, 0.1)]}))
        self.ball = self.add(Sphere(radius=0.25, label="ball", position=(-0.5, 5, 0.3),
                                    keypoints={"pole": [(0, 0, Range(0.25, 0.25))]}))


class TestKeypoints(unittest.TestCase):
    def setUp(self):
        self.world = build(Crates, 0)
        self.crate = self.world.by_name("crate")

    def test_corners_project_where_the_camera_math_says(self):
        v = view()
        points = labels.keypoints(v, self.crate, labels.keypoints_of(self.crate),
                                  np.zeros((240, 320), np.uint16), np.full((240, 320), np.inf, np.float32))
        corners = [p for p in points if p.label == "corner"]
        self.assertEqual([p.index for p in corners], list(range(8)))
        turn = labels.pose(self.crate)[1]
        for p, local in zip(corners, Box.corners((0.4, 0.6, 0.2))):
            world = np.add(self.crate.position, turn.apply(local))
            x, y = project(world, (0, 0, 0), LOOKING_NORTH, K)
            self.assertAlmostEqual(p.x, x, places=4)
            self.assertAlmostEqual(p.y, y, places=4)
            self.assertAlmostEqual(p.depth, world[1], places=4)
            self.assertTrue(p.in_frame)

    def test_a_range_in_keypoints_is_drawn(self):
        ball = self.world.by_name("ball")
        self.assertEqual(labels.keypoints_of(ball)["pole"].tolist(), [[0, 0, 0.25]])

    def test_seen_or_hidden(self):
        ids = np.zeros((240, 320), np.uint16)
        depth = np.full((240, 320), np.inf, np.float32)
        ids[100:110, 100:110], depth[100:110, 100:110] = 7, 4.0
        # On the surface, and a little behind it (it's within a few cm).
        self.assertEqual(labels.seen(105, 105, 4.0, ids, depth), (True, None))
        self.assertEqual(labels.seen(105, 105, 4.05, ids, depth), (True, None))
        # Well behind it: hidden by what's drawn there.
        self.assertEqual(labels.seen(105, 105, 5.0, ids, depth), (False, 7))
        # Just outside the outline, or right on it, with something far
        # behind (here nothing): seen. A corner on an outline falls on the
        # background half the time.
        self.assertEqual(labels.seen(110, 105, 4.0, ids, depth), (True, None))
        self.assertEqual(labels.seen(109, 105, 5.0, ids, depth), (True, None))
        # In front of everything: seen.
        self.assertEqual(labels.seen(105, 105, 1.0, ids, depth), (True, None))
        self.assertEqual(labels.seen(200, 200, 4.0, ids, depth), (True, None))

    def test_points_behind_the_camera_are_left_out_and_others_marked_out_of_frame(self):
        item = Box(size=(1, 1, 1), keypoints={"p": [(0, -10, 0), (5, 0, 0), (0, 0, 0)]})
        item.position = (0, 4, 0)
        points = labels.keypoints(view(), item, labels.keypoints_of(item), np.zeros((240, 320), np.uint16),
                                  np.full((240, 320), np.inf, np.float32))
        self.assertEqual([(p.index, p.in_frame, p.visible) for p in points], [(1, False, False), (2, True, True)])

    def test_keypoints_must_be_named_lists_of_points(self):
        class Bad(Scenario[Empty]):
            def setup(self, rng):
                self.crate = self.add(Box(keypoints=[(0, 0, 0)]))

        with self.assertRaisesRegex(ValueError, "named lists of points"):
            build(Bad, 0)

        class Flat(Scenario[Empty]):
            def setup(self, rng):
                self.crate = self.add(Box(keypoints={"corner": [(0, 0)]}))

        with self.assertRaisesRegex(ValueError, r"list of \(x, y, z\) points"):
            build(Flat, 0)


class TestFullBoxes(unittest.TestCase):
    def test_a_box_facing_the_camera(self):
        # A 1 x 1 face 4 m away: fx / 4 pixels across.
        item = Box(size=(1, 0.01, 1), position=(0, 4, 0))
        x0, y0, x1, y1, pixels = labels.full_box(view(), item)
        half = K["fx"] / 4 / 2
        self.assertAlmostEqual(x0, 160 - half, delta=0.2)
        self.assertAlmostEqual(x1, 160 + half, delta=0.2)
        self.assertAlmostEqual(y0, 120 - half, delta=0.2)
        self.assertAlmostEqual(pixels, (2 * half) ** 2, delta=4 * half)

    def test_a_sphere_is_a_little_wider_than_its_center_says(self):
        # Off to one side, a sphere's outline is an ellipse a bit wider than
        # fx * 2r / distance; the box holds all of it.
        ball = Sphere(radius=0.5, position=(0, 5, 0))
        x0, y0, x1, y1, pixels = labels.full_box(view(), ball)
        exact = 2 * K["fx"] * np.tan(np.arcsin(0.5 / 5))
        self.assertAlmostEqual(x1 - x0, exact, delta=0.3)
        self.assertAlmostEqual(y1 - y0, exact, delta=0.3)
        self.assertAlmostEqual(pixels, np.pi * (exact / 2) ** 2, delta=0.03 * pixels)

    def test_cut_to_the_picture(self):
        item = Box(size=(1, 1, 1), position=(2, 2, 0))
        x0, y0, x1, y1, _ = labels.full_box(view(), item)
        self.assertEqual(x1, 320)
        self.assertGreater(x0, 160)

    def test_a_shape_reaching_behind_the_camera(self):
        # A long box from 1 m behind the camera to 5 m ahead, below it: its
        # outline runs off the bottom of the picture.
        item = Box(size=(0.5, 6, 0.2), position=(0, 2, -0.5))
        x0, y0, x1, y1, pixels = labels.full_box(view(), item)
        self.assertEqual(y1, 240)
        self.assertGreater(y0, 120)
        self.assertGreater(pixels, 0)

    def test_none_when_out_of_the_picture(self):
        self.assertIsNone(labels.full_box(view(), Box(size=(1, 1, 1), position=(0, -4, 0))))
        self.assertIsNone(labels.full_box(view(), Box(size=(1, 1, 1), position=(30, 4, 0))))

    def test_a_turned_object(self):
        # The same box seen from another camera pose agrees with projecting
        # its corners one by one.
        item = Box(size=(0.4, 0.6, 0.2), position=(0.3, 4, 0.1), rotation=(10, 20, 30))
        eye = (1, 0, 1)
        quat = quat_looking_at(eye, item.position)
        x0, y0, x1, y1, _ = labels.full_box(view(eye, quat), item)
        corners = labels.to_world(item, Box.corners(item.size))
        pixels = np.array([project(c, eye, quat, K) for c in corners])
        np.testing.assert_allclose([x0, y0, x1, y1], [*pixels.min(0), *pixels.max(0)], atol=1e-4)

    def test_floors_and_walls_have_shapes(self):
        self.assertEqual(len(labels.shape(Floor(size=(4, 4)))[0]), 8)


if __name__ == "__main__":
    unittest.main()


class Tags(Scenario[Empty]):
    def setup(self, rng):
        # 3 m north of the camera, facing it, and one facing away.
        self.front = self.add(AprilTag(tag=5, position=(0, 3, 0), rotation=AprilTag.facing((0, -1, 0))))
        self.away = self.add(AprilTag(tag=6, position=(1, 3, 0), rotation=AprilTag.facing((0, 1, 0))))
        self.turned = self.add(AprilTag(tag=7, position=(-1, 3, 0), rotation=AprilTag.facing((-1, -1, 0))))


class TestTags(unittest.TestCase):
    def setUp(self):
        self.world = build(Tags, 0)
        self.v = view()
        self.ids = np.zeros((240, 320), np.uint16)
        self.depth = np.full((240, 320), np.inf, np.float32)

    def found(self, name):
        return labels.tag(self.v, self.world.by_name(name), self.ids, self.depth)

    def test_facing_puts_the_top_up(self):
        tag = self.world.by_name("front")
        turn = labels.pose(tag)[1]
        np.testing.assert_allclose(turn.apply([0, 0, 1]), [0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(turn.apply([0, 1, 0]), [0, 0, 1], atol=1e-9)
        # Seen from the front, its +x (the tag's right) is the viewer's right: +x here.
        np.testing.assert_allclose(turn.apply([1, 0, 0]), [1, 0, 0], atol=1e-9)

    def test_corners_go_bottom_left_bottom_right_top_right_top_left(self):
        found = self.found("front")
        c = found.corners
        self.assertEqual(found.tag, 5)
        self.assertLess(c[0, 0], c[1, 0])  # left of
        self.assertGreater(c[0, 1], c[3, 1])  # below (y down)
        self.assertAlmostEqual(c[1, 0], c[2, 0], places=3)
        self.assertGreater(c[1, 1], c[2, 1])
        self.assertLess(c[3, 0], c[2, 0])
        # 0.1651 m across at about 3 m, with fx = 160.
        self.assertAlmostEqual(c[1, 0] - c[0, 0], 160 * 0.1651 / (3 - AprilTag.THICKNESS), places=2)
        self.assertTrue(found.in_frame)
        self.assertAlmostEqual(found.angle, 0, places=4)
        self.assertAlmostEqual(found.distance, 3 - AprilTag.THICKNESS, places=4)

    def test_a_tag_turned_away_isnt_there(self):
        self.assertIsNone(self.found("away"))

    def test_angle_off_head_on(self):
        found = self.found("turned")
        to_camera = np.array([1, -3, 0]) / np.linalg.norm([1, -3, 0])
        expected = np.degrees(np.arccos(np.array([-1, -1, 0]) / np.sqrt(2) @ to_camera))
        self.assertAlmostEqual(found.angle, expected, delta=0.1)

    def test_corners_are_seen_where_the_mask_shows_the_tag(self):
        tag = self.world.by_name("front")
        self.assertEqual(self.found("front").corners_seen, 0)  # an empty mask
        c = self.found("front").corners
        x0, y0 = np.floor(c.min(0)).astype(int) - 2
        x1, y1 = np.ceil(c.max(0)).astype(int) + 2
        self.ids[y0:y1, x0:x1] = tag.id
        self.assertEqual(self.found("front").corners_seen, 4)
        # Something in front of its bottom-left corner.
        bx, by = c[0].astype(int)
        self.ids[by - 3:by + 4, bx - 3:bx + 4] = 99
        self.assertEqual(self.found("front").corners_seen, 3)
