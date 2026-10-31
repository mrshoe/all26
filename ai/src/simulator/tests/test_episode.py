"""A whole episode: logs, previews and records. A stand-in renderer draws
each labeled sphere as a disc, so these run without Blender; test_blender
checks the real one."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow.parquet as pq

from sim import Camera, Floor, Range, Scenario, Scene, Sphere
from sim import backends, records, runner
from sim.backends import Picture
from sim.backends.physics import Physics
from sim.geometry import intrinsics, project
from sim.logger import Logger
from sim.postprocess import PostProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import episodes  # noqa: E402  (ai/src/shared)


class Discs(Physics):
    """Physics, plus pictures of each sphere as a flat disc."""

    name = "discs"

    def render(self, camera, position, quat):
        k = intrinsics(camera.resolution, camera.fov)
        w, h = camera.resolution
        ids = np.zeros((h, w), np.uint16)
        ys, xs = np.mgrid[0:h, 0:w]
        for item in self.world.solids:
            if isinstance(item, Sphere):
                center = project(item.position, position, quat, k)
                if center:
                    distance = np.linalg.norm(np.subtract(item.position, position))
                    r = k["fx"] * item.radius / distance
                    ids[(xs - center[0]) ** 2 + (ys - center[1]) ** 2 <= r * r] = item.id
        rgb = np.where(ids[..., None] > 0, [250, 220, 10], [90, 70, 50]).astype(np.uint8)
        return Picture(rgb=rgb, ids=ids)


class Plain(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(20, 20)))


class Roll(Scenario[Plain]):
    duration, fps = 2.0, 10

    def setup(self, rng):
        self.ball = self.add(Sphere(radius=0.1, label="ball", position=(0, 3, 0.5), velocity=(Range(-1, 1), 0, 0)))
        self.front = self.add(Camera(position=(0, 0, 0.4), look_at=(0, 3, 0.1), resolution=(320, 240)))


def run_episode(folder, seed=4):
    info = {"scenario": "tests:Roll", "scene": "Plain", "backend": "discs"}
    holder = {}

    def make(world):
        holder["logger"] = Logger(folder, world, info, render=True)
        return holder["logger"]

    with mock.patch.object(backends, "get", lambda name: Discs()):
        runner.run(Roll, seed, "discs", make)
    episode = holder["logger"].finish()
    return episode, PostProcessor(folder, episode).run()


class TestEpisode(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def test_files(self):
        episode, previews = run_episode(self.folder)
        self.assertEqual(episode["frames"], 20)
        self.assertEqual(len(list((self.folder / "cam/front").glob("*.jpg"))), 20)
        self.assertEqual(len(list((self.folder / "mask/front").glob("*.png"))), 20)
        self.assertEqual(previews, ["preview/front.mp4", "preview/front.boxes.mp4"])
        for preview in previews:
            self.assertGreater((self.folder / preview).stat().st_size, 1000)
        saved = json.loads((self.folder / "episode.json").read_text())
        self.assertEqual(saved["seed"], 4)
        self.assertEqual(saved["labels"], {"ball": "ball"})
        self.assertEqual(saved["cameras"]["front"]["width"], 320)
        # As drawn, not where it ended up.
        self.assertEqual(saved["objects"]["ball"]["position"], [0, 3, 0.5])

    def test_signals_and_boxes(self):
        run_episode(self.folder)
        signals = pq.read_table(self.folder / "signals.parquet").to_pydict()
        self.assertEqual(signals["frame"], list(range(20)))
        self.assertAlmostEqual(signals["time"][5], 0.5)
        self.assertAlmostEqual(signals["ball.z"][0], 0.5)
        self.assertAlmostEqual(signals["ball.z"][-1], 0.1, delta=0.005)
        x0, x1 = signals["front.ball.x0"][0], signals["front.ball.x1"][0]
        self.assertLess(x0, 160)
        self.assertGreater(x1, 160)
        self.assertGreater(signals["front.ball.pixels"][0], 50)

    def test_the_same_seed_gives_the_same_data(self):
        other = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, other)
        run_episode(self.folder)
        run_episode(other)
        self.assertEqual(
            pq.read_table(self.folder / "signals.parquet").to_pydict(),
            pq.read_table(other / "signals.parquet").to_pydict(),
        )

    def test_reading_it_back(self):
        run_episode(self.folder)
        ep = episodes.Episode(self.folder)
        self.assertEqual(ep.cameras, ["front"])
        self.assertEqual(ep.frames, 20)
        self.assertEqual(ep.image("front", 3).shape, (240, 320, 3))
        self.assertEqual(ep.mask("front", 3).dtype, np.uint16)
        boxes = ep.boxes("front", 0)
        self.assertEqual([b.label for b in boxes], ["ball"])
        self.assertLess(boxes[0].x0, boxes[0].x1)

    def test_records(self):
        episode, previews = run_episode(self.folder)
        sent = []
        env = {"WILDHAT_RUN_DOC": "projects/p/databases/(default)/documents/runs/r1", "WILDHAT_EPISODE": "2"}
        with mock.patch.dict(os.environ, env), mock.patch.object(records, "patch", lambda doc, f: sent.append((doc, f))):
            records.record(episode, previews)
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
            records.record(episode, previews)
        patch.assert_not_called()

    def test_firestore_values(self):
        self.assertEqual(records.value({"a": [1, 2.5, "x", None, True]}), {"mapValue": {"fields": {"a": {"arrayValue": {"values": [
            {"integerValue": "1"}, {"doubleValue": 2.5}, {"stringValue": "x"}, {"nullValue": None}, {"booleanValue": True},
        ]}}}}})


if __name__ == "__main__":
    unittest.main()
