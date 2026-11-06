"""The scorer (shared/score.py) and the predictions writer, on a tiny
episode whose numbers can be worked out by hand."""

import json
from pathlib import Path
from unittest import mock

import numpy as np

from shared import score as scoring
from shared.datasets import Dataset
from shared.logger import Logger
from shared.predictions import Predictions
from shared.tests.test_datasets import DatasetTestCase, FakeCloud

RUN = "20261106-0900-p7tz"
TRAIN = "20261102-0019-w22f"
CAMERA, A, B = 1, 2, 3


def scored_episode(folder: Path) -> None:
    """Two frames from a 100 x 100 camera. Frame 0: ball A, box (10, 10, 30,
    30), all of it seen; ball B, box (50, 50, 60, 60), a quarter of it seen
    (full box (50, 50, 70, 70)). A has two "top" keypoints: (20, 10), seen,
    and (60, 60), hidden by A itself. Frame 1: nothing."""
    log = Logger(folder, {"scenario": "test"})
    log.add("objects", {"object": CAMERA, "name": "front", "type": "Camera", "static": False, "settings": "{}"},
            {"object": A, "name": "a", "type": "Sphere", "label": "ball", "static": False, "settings": "{}"},
            {"object": B, "name": "b", "type": "Sphere", "label": "ball", "static": False, "settings": "{}"})
    log.add("cameras", {"object": CAMERA, "width": 100, "height": 100, "fx": 50, "fy": 50, "cx": 50, "cy": 50,
                        "fov": 90})
    for frame in (0, 1):
        log.image(frame, CAMERA, "front", frame * 0.1, np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100)))
    log.add("boxes",
            {"frame": 0, "camera": CAMERA, "object": A, "label": "ball", "x0": 10, "y0": 10, "x1": 30, "y1": 30,
             "pixels": 400, "full_x0": 10, "full_y0": 10, "full_x1": 30, "full_y1": 30, "full_pixels": 400,
             "source": "truth"},
            {"frame": 0, "camera": CAMERA, "object": B, "label": "ball", "x0": 50, "y0": 50, "x1": 60, "y1": 60,
             "pixels": 100, "full_x0": 50, "full_y0": 50, "full_x1": 70, "full_y1": 70, "full_pixels": 400,
             "source": "truth"})
    log.add("keypoints",
            {"frame": 0, "camera": CAMERA, "object": A, "label": "top", "index": 0, "x": 20, "y": 10, "depth": 2,
             "in_frame": True, "visible": True, "source": "truth"},
            {"frame": 0, "camera": CAMERA, "object": A, "label": "top", "index": 1, "x": 60, "y": 60, "depth": 2.2,
             "in_frame": True, "visible": False, "occluded_by": A, "source": "truth"})
    log.finish()


class ScoreCloud(FakeCloud):
    def add_episode_files(self, prefix, index, **options):
        folder = self.root / prefix / f"episode-{index}"
        scored_episode(folder)
        for path in folder.rglob("*"):
            if path.is_file():
                self.files[f"{prefix}episode-{index}/{path.relative_to(folder).as_posix()}"] = path


class ScoreTestCase(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.cloud = ScoreCloud(self.tmp / "bucket")
        self.cloud.add_run(RUN, ["succeeded"])
        self.cloud.add_run(TRAIN, ["succeeded"])
        # RUN is held out as the test split; the scores are on it.
        self.ds = Dataset.load(self.write(f'runs = ["{TRAIN}"]\nvalidation = 0\n[test]\nruns = ["{RUN}"]\n'))
        self.ds.fetch(self.cloud, self.say)
        self.frames = {s.frame: s for s in self.ds.samples("test")}

    def predict(self, boxes=(), points=(), frames=(0, 1), classes=("ball", "top")) -> Path:
        preds = Predictions(self.tmp / "predictions", model="ball_detector", classes=list(classes))
        for frame, *box in boxes:
            preds.box(self.frames[frame], "ball", *box)
        for frame, x, y, s in points:
            preds.keypoint(self.frames[frame], "top", x, y, s)
        for frame in frames:
            preds.done(self.frames[frame])
        return preds.finish(self.ds, "test")


BOXES = [
    (0, 10, 10, 30, 30, 0.9),  # A, exactly
    (0, 11, 11, 31, 31, 0.8),  # A again: A's taken, so a false alarm
    (0, 50, 50, 60, 60, 0.7),  # B's seen part, exactly
    (0, 80, 80, 90, 90, 0.6),  # nothing there
    (1, 0, 0, 10, 10, 0.5),  # nothing there
]


class TestBoxes(ScoreTestCase):
    def test_ap_by_hand(self):
        result = scoring.score(self.predict(BOXES))
        # Highest score first: hit, false, hit, false, false, of 2 true boxes.
        # Precision 1, .5, .67, .5, .4 at recall .5, .5, 1, 1, 1; made
        # non-increasing: 1 up to recall .5 (51 of 101 points), .67 after.
        expected = (51 * 1 + 50 * 2 / 3) / 101
        ball = result["boxes"]["classes"]["ball"]
        self.assertAlmostEqual(ball["ap50"], expected, places=4)
        self.assertEqual((ball["truth"], ball["predicted"], ball["recall50"]), (2, 5, 1.0))
        # The two hits are exact, so they hit at every IoU up to 0.95.
        self.assertAlmostEqual(ball["ap"], expected, places=4)
        self.assertEqual(ball["best_f1"], {"threshold": 0.7, "f1": 0.8, "precision": 0.6667, "recall": 1.0})
        self.assertEqual(result["pictures"], 2)

    def test_ignoring_mostly_hidden_objects(self):
        # B is a quarter seen: ignored, so finding it isn't a hit, and
        # there's one true box: the 0.9 hits it, the rest are false.
        result = scoring.score(self.predict(BOXES), min_share=0.3)
        self.assertEqual(result["boxes"]["classes"]["ball"]["truth"], 1)
        self.assertAlmostEqual(result["boxes"]["ap50"], 1.0)
        # Ignoring by pixels: B has 100.
        self.assertEqual(scoring.score(self.predict(BOXES), min_pixels=101)["boxes"]["classes"]["ball"]["truth"], 1)

    def test_full_boxes(self):
        # Against B's full box (50, 50, 70, 70), the prediction of its seen
        # part overlaps a quarter: no longer a hit at IoU 0.5.
        visible = scoring.score(self.predict(BOXES))["boxes"]["ap50"]
        full = scoring.score(self.predict(BOXES), boxes="full")["boxes"]["ap50"]
        self.assertLess(full, visible)
        found = scoring.score(self.predict([(0, 10, 10, 30, 30, 0.9), (0, 50, 50, 70, 70, 0.8)]), boxes="full")
        self.assertAlmostEqual(found["boxes"]["ap50"], 1.0)

    def test_tighter_boxes_score_better_ap(self):
        # A box off by 2 px is a hit at IoU 0.5 but not at 0.95.
        loose = scoring.score(self.predict([(0, 12, 12, 32, 32, 0.9), (0, 50, 50, 60, 60, 0.8)]))["boxes"]
        self.assertAlmostEqual(loose["ap50"], 1.0)
        self.assertLess(loose["ap"], 1.0)

    def test_by_size(self):
        # A is 400 px, B 100: both small (under 32 x 32).
        sizes = scoring.score(self.predict(BOXES))["boxes"]["by_size"]
        self.assertEqual({k: v["truth"] for k, v in sizes.items()}, {"small": 2, "medium": 0, "large": 0})
        self.assertIsNone(sizes["large"]["ap50"])

    def test_nothing_predicted_scores_zero(self):
        result = scoring.score(self.predict([]))
        self.assertEqual(result["boxes"]["ap50"], 0.0)
        self.assertEqual(result["keypoints"]["found@4"], 0.0)

    def test_a_box_detector_isnt_scored_on_keypoints(self):
        self.assertNotIn("keypoints", scoring.score(self.predict(BOXES)))

    def test_only_the_models_classes(self):
        # A model of something else: the truth's balls aren't its business.
        result = scoring.score(self.predict([(0, 0, 0, 10, 10, 0.5)], classes=["fuel"]))
        self.assertNotIn("boxes", result)
        # Left out, its classes are the ones it predicted.
        preds = Predictions(self.tmp / "unsaid", model="m")
        preds.box(self.frames[0], "ball", 10, 10, 30, 30, 0.9)
        info = json.loads((preds.finish(self.ds, "test") / "predictions.json").read_text())
        self.assertEqual(info["classes"], ["ball"])

    def test_metrics_for_a_training_run(self):
        flat = scoring.metrics(scoring.score(self.predict(BOXES)), "test")
        self.assertEqual(sorted(flat), ["test_box_ap", "test_box_ap50", "test_box_ap75"])


class TestKeypoints(ScoreTestCase):
    POINTS = [
        (0, 21, 10, 0.9),  # 1 px from the seen point
        (0, 60, 63, 0.8),  # 3 px from the hidden one
        (0, 5, 90, 0.3),  # nothing near
    ]

    def test_seen_points(self):
        k = scoring.score(self.predict(points=self.POINTS))["keypoints"]
        top = k["classes"]["top"]
        self.assertEqual(top["truth"], 1)  # the hidden one doesn't count
        for r in (2, 4, 8):
            self.assertEqual(top[f"found@{r}"], 1.0)
            self.assertAlmostEqual(top[f"ap@{r}"], 1.0)
        # Within 2 px, the point near the hidden one is false too (2 false
        # points over 2 pictures); within 4, it's on an ignored point.
        self.assertEqual(k["false_per_picture@2"], 1.0)
        self.assertEqual(k["false_per_picture@4"], 0.5)

    def test_by_visibility(self):
        k = scoring.score(self.predict(points=self.POINTS))["keypoints"]["by_visibility"]
        self.assertEqual(k[scoring.SEEN], {"truth": 1, "found@2": 1.0, "found@4": 1.0, "found@8": 1.0})
        self.assertEqual(k[scoring.SELF], {"truth": 1, "found@2": 0.0, "found@4": 1.0, "found@8": 1.0})
        self.assertEqual(k[scoring.OTHER]["truth"], 0)

    def test_all_points(self):
        k = scoring.score(self.predict(points=self.POINTS), all_points=True)["keypoints"]["classes"]["top"]
        self.assertEqual(k["truth"], 2)
        self.assertEqual((k["found@2"], k["found@4"]), (0.5, 1.0))


class TestPredictionsAndCommandLine(ScoreTestCase):
    def test_the_folder(self):
        folder = self.predict(BOXES, TestKeypoints.POINTS)
        info = json.loads((folder / "predictions.json").read_text())
        self.assertEqual((info["model"], info["split"], info["boxes"], info["keypoints"]), ("ball_detector", "test", 5, 3))
        self.assertEqual(info["pictures"], [[RUN, 0, CAMERA, 0], [RUN, 0, CAMERA, 1]])
        self.assertTrue((folder / RUN / "episode-0" / "boxes.parquet").exists())

    def test_mistakes(self):
        preds = Predictions(self.tmp / "p", model="m")
        with self.assertRaisesRegex(ValueError, "past its x0"):
            preds.box(self.frames[0], "ball", 10, 10, 5, 20, 0.9)
        with self.assertRaisesRegex(ValueError, "needs numbers"):
            preds.box(self.frames[0], "ball", 0, 0, float("nan"), 1, 0.9)
        with self.assertRaisesRegex(ValueError, "Name the model"):
            Predictions(self.tmp / "p", model="truth")
        with self.assertRaisesRegex(ValueError, "no predictions.json"):
            scoring.score(self.tmp / "nowhere")
        # Predictions on the test split, scored against validation (empty).
        with self.assertRaisesRegex(ValueError, "no pictures"):
            scoring.score(self.predict(BOXES), split="validation")

    def test_command_line(self):
        folder = self.predict(BOXES, TestKeypoints.POINTS)
        with mock.patch("builtins.print") as printed:
            self.assertEqual(scoring.main([str(folder)]), 0)
        text = "\n".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("ball_detector on", text)
        self.assertIn("Boxes (visible)", text)
        self.assertIn("Keypoints (seen)", text)
        saved = json.loads((folder / "score.json").read_text())
        self.assertAlmostEqual(saved["boxes"]["ap50"], (51 + 50 * 2 / 3) / 101, places=4)
