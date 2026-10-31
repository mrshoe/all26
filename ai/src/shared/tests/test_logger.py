"""The shared Logger as the robot will use it: no simulation, no masks, and
signals that come and go. Run from ai/src:

    python -m unittest discover -s shared/tests -t .
"""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shared.episodes import Episode
from shared.logger import Logger
from shared.postprocess import PostProcessor


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def test_a_robot_episode(self):
        log = Logger(self.folder, {"robot": "kai-bot", "fps": 5})
        image = np.full((48, 64, 3), 120, np.uint8)
        log.frame(0, 0.0, {"model.ms": 11.0}, images={"front": image})
        log.frame(1, 0.21, {"model.ms": 12.5, "front.ball.x0": 10, "front.ball.y0": 8,
                            "front.ball.x1": 20, "front.ball.y1": 18, "front.ball.pixels": 100},
                  images={"front": image})
        log.frame(2, 0.4, {}, images={"front": image})
        info = log.finish({"cameras": {"front": {}}, "labels": {"ball": "ball"}})

        self.assertEqual(info["frames"], 3)
        self.assertEqual(info["robot"], "kai-bot")
        on_disk = json.loads((self.folder / "episode.json").read_text())
        self.assertEqual(on_disk, info)

        ep = Episode(self.folder)
        self.assertEqual(ep.signals["frame"], [0, 1, 2])
        self.assertEqual(ep.signals["model.ms"][:2], [11.0, 12.5])
        self.assertTrue(math.isnan(ep.signals["model.ms"][2]))  # missing that frame
        self.assertTrue(math.isnan(ep.signals["front.ball.x0"][0]))  # not seen yet
        self.assertEqual([b.name for b in ep.boxes("front", 1)], ["ball"])
        self.assertEqual(ep.boxes("front", 0), [])
        self.assertEqual(ep.image("front", 2).shape, (48, 64, 3))
        self.assertFalse((self.folder / "mask").exists())

        made = PostProcessor(self.folder, info).run()
        self.assertEqual(made, ["preview/front.mp4", "preview/front.boxes.mp4"])

    def test_numpy_values_in_info(self):
        info = Logger(self.folder).finish({"size": np.array([1, 2]), "k": np.float32(0.5)})
        self.assertEqual(info["size"], [1, 2])
        self.assertEqual(info["k"], 0.5)


if __name__ == "__main__":
    unittest.main()
