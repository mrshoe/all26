"""The shared Logger as the robot will use it: no simulation, no masks, its
own per-frame table, and a model's detections. Run from ai/src:

    python -m unittest discover -s shared/tests -t .
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from shared.episodes import Episode
from shared.logger import Logger, schema
from shared.postprocess import PostProcessor

ROBOT = schema(("frame", pa.int32()), ("time", pa.float64()), ("loop_ms", pa.float64()), key=("frame",))


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def test_a_robot_episode(self):
        log = Logger(self.folder, {"robot": "kai-bot", "fps": 5}, schemas={"robot": ROBOT})
        log.add("objects", {"object": 1, "name": "front", "type": "Camera", "static": False, "settings": "{}"})
        log.add("cameras", {"object": 1, "width": 64, "height": 48, "fx": 50, "fy": 50, "cx": 32, "cy": 24, "fov": 65})
        image = np.full((48, 64, 3), 120, np.uint8)
        for frame in range(3):
            log.add("robot", {"frame": frame, "time": frame * 0.2, "loop_ms": 11.0 + frame})
            log.image(frame, 1, "front", frame * 0.2 + 0.01, image)
        log.add("boxes", {"frame": 1, "camera": 1, "label": "ball", "x0": 10, "y0": 8, "x1": 20, "y1": 18,
                          "score": 0.9, "source": "yolo-ball-v3"})
        info = log.finish()

        self.assertEqual(info["frames"], 3)
        self.assertEqual(info["tables"], ["objects", "cameras", "robot", "images", "boxes"])
        self.assertEqual(json.loads((self.folder / "episode.json").read_text()), info)
        self.assertEqual(pq.read_table(self.folder / "robot.parquet")["loop_ms"].to_pylist(), [11.0, 12.0, 13.0])

        ep = Episode(self.folder)
        self.assertEqual(ep.cameras, ["front"])
        self.assertEqual(ep.image("front", 2).shape, (48, 64, 3))
        with self.assertRaises(KeyError):
            ep.mask("front", 2)  # real cameras have none
        self.assertEqual(ep.boxes("front", 1), [])  # no ground truth
        found = ep.boxes("front", 1, source="yolo-ball-v3")
        self.assertEqual([(b.name, b.label, round(b.score, 2)) for b in found], [(None, "ball", 0.9)])

        made = PostProcessor(self.folder).run()
        self.assertEqual(made, ["preview/front.mp4"])

    def test_rows_are_checked(self):
        log = Logger(self.folder)
        with self.assertRaisesRegex(KeyError, "no column speed"):
            log.add("sim", {"frame": 0, "time": 0, "speed": 3})
        with self.assertRaisesRegex(ValueError, "needs render_s"):
            log.add("sim", {"frame": 0, "time": 0, "step_s": 0, "physics_s": 0, "log_s": 0})
        with self.assertRaisesRegex(KeyError, "no table"):
            log.add("signals", {"frame": 0})

    def test_rows_are_sorted_by_key(self):
        log = Logger(self.folder)
        for frame, obj in [(1, 2), (0, 2), (1, 1), (0, 1)]:
            log.add("poses", {"frame": frame, "object": obj, **dict.fromkeys(
                ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"], 0.0)})
        log.finish()
        t = pq.read_table(self.folder / "poses.parquet")
        self.assertEqual(list(zip(t["frame"].to_pylist(), t["object"].to_pylist())), [(0, 1), (0, 2), (1, 1), (1, 2)])

    def test_numpy_values_in_info(self):
        info = Logger(self.folder).finish({"size": np.array([1, 2]), "k": np.float32(0.5)})
        self.assertEqual(info["size"], [1, 2])
        self.assertEqual(info["k"], 0.5)

    def test_old_episodes_are_refused(self):
        (self.folder / "episode.json").write_text('{"format": 1}')
        with self.assertRaisesRegex(ValueError, "old format"):
            Episode(self.folder)


if __name__ == "__main__":
    unittest.main()
