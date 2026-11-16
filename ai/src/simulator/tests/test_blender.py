"""The Blender backend renders what physics says, with exact masks and
depth, so keypoints and full boxes agree with what's drawn. Skipped
without bpy (pip install bpy==4.5.4, Python 3.11)."""

import unittest

import numpy as np

from sim import AprilTag, Box, Camera, Floor, Light, Scenario, Scene, Sphere
from sim import labels
from sim.geometry import intrinsics, project
from sim.logger import boxes as logger_boxes
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


CRATE = (0.5, 0.5, 0.5)


class Hiding(Scenario[Lit]):
    """A crate with its corners as keypoints, a block hiding its near top
    left corner, and a ball half behind a post."""

    hidden = True

    def setup(self, rng):
        self.crate = self.add(Box(size=CRATE, position=(-1, 4, 0.25), color=(0.2, 0.3, 0.8),
                                  keypoints={"corner": Box.corners(CRATE)}))
        self.ball = self.add(Sphere(radius=0.2, label="ball", color=(1, 0.85, 0), position=(0.4, 3, 0.2),
                                    static=True))
        if self.hidden:
            # Halfway between the camera and the crate's near top left corner.
            self.block = self.add(Box(size=(0.2, 0.2, 0.2), position=(-0.625, 1.875, 0.55)))
            # In front of the ball's left half.
            self.post = self.add(Box(size=(0.2, 0.05, 1), position=(0.3, 2, 0.5)))
        self.cam = self.add(Camera(position=(0, 0, 0.6), look_at=(0, 3, 0.2), resolution=(320, 240), fov=60))


class Unhidden(Hiding):
    hidden = False


def render(scenario):
    world = build(scenario, 0)
    backend = Blender(samples=4)
    backend.build(world)
    read_poses(world, backend)
    cam = world.cameras[0]
    pose = camera_pose(cam)
    try:
        return world, pose, backend.render(cam, *pose)
    finally:
        backend.close()


@unittest.skipIf(Blender is None, "bpy isn't installed")
class TestKeypointsAndFullBoxes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, pose, cls.picture = render(Hiding)
        cls.view = labels.View(*pose, intrinsics((320, 240), 60))
        cls.unhidden_world, _, cls.unhidden = render(Unhidden)
        crate = cls.world.by_name("crate")
        cls.corners = labels.keypoints(cls.view, crate, labels.keypoints_of(crate), cls.picture.ids,
                                       cls.picture.depth)

    def test_depth_is_along_the_cameras_axis(self):
        # The crate's front face is 3.75 m ahead of the camera, along y; the
        # camera looks a little down, so along its axis a little less.
        crate = self.world.by_name("crate")
        front = self.view.local([(-1, 3.75, 0.25)])
        ys, xs = np.nonzero(self.picture.ids == crate.id)
        self.assertTrue(np.isinf(self.picture.depth[self.picture.ids == 0]).all())
        near = self.picture.depth[ys, xs].min()
        self.assertAlmostEqual(near, -front[0, 2], delta=0.15)

    def test_the_corners_are_on_the_crate(self):
        crate = self.world.by_name("crate")
        for p in self.corners:
            if p.visible:
                row, col = int(p.y), int(p.x)
                around = self.unhidden.ids[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
                self.assertIn(crate.id, around, p)

    def test_which_corners_are_seen(self):
        crate, block = self.world.by_name("crate"), self.world.by_name("block")
        by = {p.index: (p.visible, p.occluded_by) for p in self.corners}
        # Corner i is at -x or +x as bit 2 of i is 0 or 1, y bit 1, z bit 0.
        self.assertEqual(by[0b101], (True, None))  # near, top, right: facing the camera
        self.assertEqual(by[0b001], (False, block.id))  # near, top, left: behind the block
        self.assertEqual(by[0b010], (False, crate.id))  # far, bottom, left: behind the crate itself

    def test_the_full_box_of_a_half_hidden_ball_is_its_unhidden_box(self):
        ball = self.world.by_name("ball")
        seen = logger_boxes(self.picture.ids, [ball])[ball]
        whole = logger_boxes(self.unhidden.ids, [self.unhidden_world.by_name("ball")])
        (x0, y0, x1, y1, pixels), = whole.values()
        fx0, fy0, fx1, fy1, full_pixels = labels.full_box(self.view, ball)
        np.testing.assert_allclose([fx0, fy0, fx1, fy1], [x0, y0, x1, y1], atol=1.5)
        self.assertAlmostEqual(full_pixels / pixels, 1, delta=0.05)
        self.assertLess(seen[4] / full_pixels, 0.8)  # the post hides some
        self.assertLess(seen[2] - seen[0], x1 - x0)


try:
    import cv2
except ImportError:
    cv2 = None


class Tagged(Scenario[Lit]):
    def setup(self, rng):
        self.wall = self.add(Box(size=(3, 0.1, 2), position=(0, 3.05, 1), color=(0.5, 0.5, 0.5)))
        self.head_on = self.add(AprilTag(tag=1, position=(-0.4, 3.0, 1.0), rotation=AprilTag.facing((0, -1, 0))))
        self.turned = self.add(AprilTag(tag=7, position=(0.5, 2.7, 0.9), rotation=AprilTag.facing((-0.5, -1, 0.2))))
        self.away = self.add(AprilTag(tag=12, position=(0, 2.0, 0.5), rotation=AprilTag.facing((0, 1, 0))))
        self.cam = self.add(Camera(position=(0.1, 1.0, 1.1), look_at=(0, 3, 1.0), resolution=(640, 480), fov=60))


@unittest.skipIf(Blender is None or cv2 is None, "bpy or OpenCV (opencv-python-headless) isn't installed")
class TestAprilTags(unittest.TestCase):
    """Rendered tags are real 36h11 tags: OpenCV's detector reads their
    IDs, and finds their corners where the labels say, in the same order."""

    @classmethod
    def setUpClass(cls):
        cls.world = build(Tagged, 0)
        backend = Blender(samples=8)
        backend.build(cls.world)
        read_poses(cls.world, backend)
        cam = cls.world.cameras[0]
        pose = camera_pose(cam)
        cls.picture = backend.render(cam, *pose)
        backend.close()
        view = labels.View(*pose, intrinsics(cam.resolution, cam.fov))
        cls.truth = {t.name: labels.tag(view, t, cls.picture.ids, cls.picture.depth)
                     for t in cls.world.of_type(AprilTag)}

    def test_only_tags_facing_the_camera_are_labeled(self):
        self.assertIsNone(self.truth["away"])
        self.assertEqual(self.truth["head_on"].corners_seen, 4)
        self.assertEqual(self.truth["turned"].corners_seen, 4)

    def test_opencv_finds_them_where_the_labels_say(self):
        from shared.apriltags import detect

        found = {tag: corners for tag, corners in detect(self.picture.rgb)}
        self.assertEqual(set(found), {1, 7})
        for name in ("head_on", "turned"):
            truth = self.truth[name]
            error = np.linalg.norm(found[truth.tag] - truth.corners, axis=1)
            self.assertLess(error.max(), 1.0, f"{name}: corners {error.round(2)} px off")

    def test_the_mask_holds_the_tags(self):
        ids = set(np.unique(self.picture.ids).tolist())
        self.assertIn(self.world.by_name("head_on").id, ids)


class Gray(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(10, 10), color=(0.4, 0.4, 0.4)))
        self.sun = self.add(Light(kind="sun", intensity=4, direction=(0.3, 0.5, -1)))


class Passing(Scenario[Gray]):
    fps = 10

    def setup(self, rng):
        self.ball = self.add(Sphere(radius=0.15, color=(1, 0.85, 0), position=(-1, 3, 0.15), velocity=(8, 0, 0)))
        self.sharp = self.add(Camera(position=(0, 0, 0.6), look_at=(0, 3, 0.2), resolution=(160, 120), fov=60))
        self.blurred = self.add(Camera(position=(0, 0, 0.6), look_at=(0, 3, 0.2), resolution=(160, 120), fov=60,
                                       shutter=0.05))


@unittest.skipIf(Blender is None, "bpy isn't installed")
class TestMotionBlur(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        world = build(Passing, 0)
        backend = Blender(samples=8)
        backend.build(world)
        cls.pictures = []
        # Two frames, so the second has a frame before it to blur from.
        for frame in range(2):
            if frame:
                backend.step(0.1)
            read_poses(world, backend)
            cls.pictures.append({cam.name: backend.render(cam, *camera_pose(cam)) for cam in world.cameras})
        backend.close()
        cls.ball = world.by_name("ball")

    def yellow_width(self, rgb):
        r, g, b = (rgb[..., i].astype(float) for i in range(3))
        cols = np.nonzero(((r - b > 60) & (g - b > 40)).any(0))[0]
        return cols.max() - cols.min() + 1 if len(cols) else 0

    def test_a_shutter_blurs_what_moves(self):
        now = self.pictures[1]
        sharp, blurred = self.yellow_width(now["sharp"].rgb), self.yellow_width(now["blurred"].rgb)
        # 8 m/s for 0.05 s is 0.4 m, more than the ball's width again.
        self.assertGreater(blurred, sharp * 1.5)
        # The first frame has nothing before it: no blur.
        first = self.pictures[0]
        self.assertAlmostEqual(self.yellow_width(first["blurred"].rgb), self.yellow_width(first["sharp"].rgb), delta=2)

    def test_the_labels_are_where_it_is_when_the_shutter_closes(self):
        sharp, blurred = self.pictures[1]["sharp"].ids, self.pictures[1]["blurred"].ids
        xs = np.nonzero(sharp == self.ball.id)[1]
        self.assertGreater(len(xs), 0)
        # The blurred mask is from somewhere in the shutter's time: never ahead of the sharp one.
        bxs = np.nonzero(blurred == self.ball.id)[1]
        self.assertLessEqual(bxs.max(), xs.max() + 1)
