"""A whole episode: logs, previews and records. A stand-in renderer draws
each labeled sphere as a flat disc facing the camera, so these run without
Blender; test_blender checks the real one."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow.parquet as pq

from sim import Camera, Floor, Range, Scenario, Scene, Sphere
from sim import backends, labels, records, runner
from sim.backends import Picture
from sim.backends.physics import Physics
from sim.geometry import intrinsics, project
from sim.logger import SimLogger

from shared import episodes  # noqa: E402  (ai/src/shared, on the path once sim is imported)
from shared.postprocess import PostProcessor  # noqa: E402


class Discs(Physics):
    """Physics, plus pictures of each sphere as a flat disc."""

    name = "discs"

    def render(self, camera, position, quat):
        k = intrinsics(camera.resolution, camera.fov)
        w, h = camera.resolution
        ids = np.zeros((h, w), np.uint16)
        depth = np.full((h, w), np.inf, np.float32)
        ys, xs = np.mgrid[0:h, 0:w]
        for item in self.world.solids:
            if isinstance(item, Sphere):
                center = project(item.position, position, quat, k)
                if center:
                    distance = np.linalg.norm(np.subtract(item.position, position))
                    r = k["fx"] * item.radius / distance
                    disc = (xs - center[0]) ** 2 + (ys - center[1]) ** 2 <= r * r
                    ids[disc] = item.id
                    # The disc is at the sphere's center (along the camera's axis).
                    depth[disc] = labels.View(position, quat, k).local([item.position])[0, 2] * -1
        rgb = np.where(ids[..., None] > 0, [250, 220, 10], [90, 70, 50]).astype(np.uint8)
        return Picture(rgb=rgb, ids=ids, depth=depth)


class Plain(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(20, 20)))


class Roll(Scenario[Plain]):
    duration, fps = 2.0, 10

    def setup(self, rng):
        # Its center, and points 2 cm in front of and behind it: the disc
        # hides the one behind.
        self.ball = self.add(Sphere(radius=0.1, label="ball", position=(0, 3, 0.5), velocity=(Range(-1, 1), 0, 0),
                                    keypoints={"center": [(0, 0, 0)], "pin": [(0, -0.2, 0), (0, 0.2, 0)]}))
        self.front = self.add(Camera(position=(0, 0, 0.4), look_at=(0, 3, 0.1), resolution=(320, 240)))


def run_episode(folder, seed=4):
    info = {"scenario": "tests:Roll", "scene": "Plain", "backend": "discs"}
    holder = {}

    def make(world):
        holder["logger"] = SimLogger(folder, world, info, render=True)
        return holder["logger"]

    with mock.patch.object(backends, "get", lambda name: Discs()):
        runner.run(Roll, seed, "discs", make)
    episode = holder["logger"].finish()
    return episode, PostProcessor(folder).run()


class TestEpisode(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def test_files(self):
        episode, previews = run_episode(self.folder)
        self.assertEqual(episode["frames"], 20)
        self.assertEqual(len(list((self.folder / "cam/front").glob("*.jpg"))), 20)
        self.assertEqual(len(list((self.folder / "mask/front").glob("*.png"))), 20)
        self.assertEqual(previews, ["preview/front.mp4"])
        for preview in previews:
            self.assertGreater((self.folder / preview).stat().st_size, 1000)
        saved = json.loads((self.folder / "episode.json").read_text())
        self.assertEqual(saved["format"], 2)
        self.assertEqual(saved["seed"], 4)
        self.assertEqual(saved["tables"], ["objects", "cameras", "poses", "images", "boxes", "keypoints", "sim"])
        for table in saved["tables"]:
            self.assertTrue((self.folder / f"{table}.parquet").exists())

    def test_objects_and_cameras(self):
        run_episode(self.folder)
        objects = {r["name"]: r for r in pq.read_table(self.folder / "objects.parquet").to_pylist()}
        self.assertEqual(sorted(objects), ["ball", "floor", "front"])
        ball = objects["ball"]
        self.assertEqual((ball["type"], ball["label"], ball["static"]), ("Sphere", "ball", False))
        self.assertTrue(objects["floor"]["static"])
        self.assertFalse(objects["front"]["static"])
        # As drawn, not where it ended up.
        self.assertEqual(json.loads(ball["settings"])["position"], [0, 3, 0.5])
        cameras = pq.read_table(self.folder / "cameras.parquet").to_pylist()
        self.assertEqual(cameras[0]["object"], objects["front"]["object"])
        self.assertEqual((cameras[0]["width"], cameras[0]["height"]), (320, 240))

    def test_poses(self):
        run_episode(self.folder)
        ids = {r["name"]: r["object"] for r in pq.read_table(self.folder / "objects.parquet").to_pylist()}
        poses = pq.read_table(self.folder / "poses.parquet").to_pylist()
        ball = [r for r in poses if r["object"] == ids["ball"]]
        self.assertEqual([r["frame"] for r in ball], list(range(20)))
        self.assertAlmostEqual(ball[0]["z"], 0.5)
        self.assertAlmostEqual(ball[-1]["z"], 0.1, delta=0.005)
        # Static: frame 0 only.
        self.assertEqual([r["frame"] for r in poses if r["object"] == ids["floor"]], [0])
        # The camera doesn't move, so its velocities are 0.
        front = [r for r in poses if r["object"] == ids["front"]]
        self.assertEqual(len(front), 20)
        self.assertEqual(max(abs(r["vx"]) + abs(r["wz"]) for r in front), 0)

    def test_rolling_spins(self):
        run_episode(self.folder)
        ids = {r["name"]: r["object"] for r in pq.read_table(self.folder / "objects.parquet").to_pylist()}
        last = [r for r in pq.read_table(self.folder / "poses.parquet").to_pylist() if r["object"] == ids["ball"]][-1]
        # Rolling along x on the floor: spinning about y at speed / radius.
        self.assertAlmostEqual(last["wy"], last["vx"] / 0.1, delta=0.05 * abs(last["vx"] / 0.1) + 0.05)

    def test_sim_and_images(self):
        run_episode(self.folder)
        sim = pq.read_table(self.folder / "sim.parquet").to_pydict()
        self.assertEqual(sim["frame"], list(range(20)))
        self.assertAlmostEqual(sim["time"][5], 0.5)
        self.assertGreater(sum(sim["physics_s"]), 0)
        self.assertGreater(sum(sim["render_s"]), 0)
        self.assertEqual(sim["model_ms"], [None] * 20)
        images = pq.read_table(self.folder / "images.parquet").to_pylist()
        self.assertEqual(images[3]["image"], "cam/front/000003.jpg")
        self.assertEqual(images[3]["mask"], "mask/front/000003.png")

    def test_boxes(self):
        run_episode(self.folder)
        boxes = pq.read_table(self.folder / "boxes.parquet").to_pylist()
        first = boxes[0]
        self.assertEqual((first["frame"], first["label"], first["source"], first["score"]), (0, "ball", "truth", None))
        self.assertLess(first["x0"], 160)
        self.assertGreater(first["x1"], 160)
        self.assertGreater(first["pixels"], 50)

    def test_the_same_seed_gives_the_same_data(self):
        other = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, other)
        run_episode(self.folder)
        run_episode(other)
        for table in ("objects", "cameras", "poses", "images", "boxes", "keypoints"):
            self.assertEqual(pq.read_table(self.folder / f"{table}.parquet").to_pydict(),
                             pq.read_table(other / f"{table}.parquet").to_pydict(), table)
        # The sim table's timings differ, but not its times.
        self.assertEqual(pq.read_table(self.folder / "sim.parquet")["time"].to_pylist(),
                         pq.read_table(other / "sim.parquet")["time"].to_pylist())

    def test_reading_it_back(self):
        run_episode(self.folder)
        ep = episodes.Episode(self.folder)
        self.assertEqual(ep.cameras, ["front"])
        self.assertEqual(ep.frames, 20)
        self.assertEqual(ep.image("front", 3).shape, (240, 320, 3))
        self.assertEqual(ep.mask("front", 3).dtype, np.uint16)
        boxes = ep.boxes("front", 0)
        self.assertEqual([(b.label, b.name) for b in boxes], [("ball", "ball")])
        self.assertLess(boxes[0].x0, boxes[0].x1)
        self.assertEqual(ep.objects["ball"]["settings"]["radius"], 0.1)
        self.assertEqual(ep.boxes("front", 0, source="some-model"), [])

    def test_full_boxes(self):
        run_episode(self.folder)
        ep = episodes.Episode(self.folder)
        # Near the middle of the picture, nothing hides the ball, so its full
        # box is its box.
        (box,) = ep.boxes("front", 0)
        np.testing.assert_allclose(box.full, (box.x0, box.y0, box.x1, box.y1), atol=1)
        self.assertAlmostEqual(box.share, 1, delta=0.05)
        # Off to the side, a sphere's outline stretches, but the stand-in's
        # discs don't (test_blender checks real outlines), so only: it holds
        # the box seen.
        (box,) = ep.boxes("front", 19)
        x0, y0, x1, y1 = box.full
        self.assertTrue(x0 <= box.x0 and y0 <= box.y0 and x1 >= box.x1 and y1 >= box.y1)

    def test_keypoints(self):
        run_episode(self.folder)
        ep = episodes.Episode(self.folder)
        points = ep.keypoints("front", 0)
        self.assertEqual([(p.name, p.label, p.index) for p in points],
                         [("ball", "center", 0), ("ball", "pin", 0), ("ball", "pin", 1)])
        center, near, far = points
        self.assertTrue(center.visible and center.in_frame)
        self.assertEqual(center.occluded_by, None)
        (box,) = ep.boxes("front", 0)
        self.assertAlmostEqual(center.x, (box.x0 + box.x1) / 2, delta=1)
        # 20 cm toward the camera: nothing in front of it.
        self.assertEqual((near.visible, near.occluded_by), (True, None))
        # 20 cm behind: the ball's own disc hides it.
        self.assertEqual((far.visible, far.occluded_by), (False, "self"))
        self.assertGreater(far.depth, center.depth)
        self.assertEqual(ep.keypoints("front", 0, source="some-model"), [])

    def test_records(self):
        episode, previews = run_episode(self.folder)
        sent = []
        env = {"WILDHAT_RUN_DOC": "projects/p/databases/(default)/documents/runs/r1", "WILDHAT_EPISODE": "2"}
        with mock.patch.dict(os.environ, env), mock.patch.object(records, "patch", lambda doc, f: sent.append((doc, f))):
            records.record(episode, previews, ["front"], ["ball"])
        (run_doc, run), (ep_doc, mine) = sent
        self.assertEqual(run_doc, env["WILDHAT_RUN_DOC"])
        self.assertEqual(run, {"sim": {"scenario": "tests:Roll", "scene": "Plain", "backend": "discs"}})
        self.assertEqual(ep_doc, env["WILDHAT_RUN_DOC"] + "/episodes/2")
        self.assertEqual(mine["sim"]["frames"], 20)
        self.assertEqual(mine["sim"]["labels"], ["ball"])
        self.assertEqual(mine["sim"]["previews"], previews)

    def test_no_records_outside_the_cloud(self):
        episode, previews = run_episode(self.folder)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(records, "patch") as patch:
            records.record(episode, previews, ["front"], ["ball"])
        patch.assert_not_called()

    def test_firestore_values(self):
        self.assertEqual(records.value({"a": [1, 2.5, "x", None, True]}), {"mapValue": {"fields": {"a": {"arrayValue": {"values": [
            {"integerValue": "1"}, {"doubleValue": 2.5}, {"stringValue": "x"}, {"nullValue": None}, {"booleanValue": True},
        ]}}}}})


if __name__ == "__main__":
    unittest.main()
