"""Datasets: reading dataset.toml, the split, fetching runs into the cache, and
the samples. The cloud is faked from real episode folders the Logger writes.
Run from ai/src:

    python -m unittest discover -s shared/tests -t .
"""

import os
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from shared import datasets
from shared.cloud import CloudError
from shared.datasets import Dataset, _yolo_line, split_share
from shared.logger import Logger

BUCKET = "test-project-wildhat"


def episode_folder(folder: Path, frames=4, cameras=("front",), boxes_per_frame=1, box=(1, 1, 5, 5),
                   full=None) -> None:
    """A small simulated episode: cameras, a labeled ball, a box in every
    frame of every camera, and a preview video. `full` is the full box
    (x0, y0, x1, y1, pixels), for episodes simulated since there were any."""
    log = Logger(folder, {"scenario": "test"})
    ids = {}
    for i, cam in enumerate(cameras, start=1):
        ids[cam] = i
        log.add("objects", {"object": i, "name": cam, "type": "Camera", "static": False, "settings": "{}"})
        log.add("cameras", {"object": i, "width": 32, "height": 24, "fx": 30, "fy": 30, "cx": 16, "cy": 12, "fov": 60})
    ball = len(cameras) + 1
    log.add("objects", {"object": ball, "name": "ball", "type": "Sphere", "label": "ball", "static": False,
                        "settings": "{}"})
    image = np.zeros((24, 32, 3), np.uint8)
    mask = np.zeros((24, 32), np.uint16)
    for frame in range(frames):
        for cam, cid in ids.items():
            log.image(frame, cid, cam, frame * 0.1, image, mask)
            for _ in range(boxes_per_frame):
                row = {"frame": frame, "camera": cid, "object": ball, "label": "ball",
                       "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                       "pixels": (box[2] - box[0]) * (box[3] - box[1]), "source": "truth"}
                if full:
                    row.update(zip(("full_x0", "full_y0", "full_x1", "full_y1", "full_pixels"), full))
                log.add("boxes", row)
    log.finish()
    (folder / "preview").mkdir()
    (folder / "preview" / "front.mp4").write_bytes(b"video")


class FakeCloud:
    """Runs' records and output files, the way launch and the base image's
    entrypoint leave them."""

    def __init__(self, root: Path):
        self.root = root
        self.docs: dict[str, dict] = {}
        self.files: dict[str, Path] = {}
        self.downloads: list[str] = []

    def add_run(self, run: str, statuses: list[str], who="kai-morales", **episode_options) -> None:
        prefix = f"runs/{who}/{run}/"
        self.docs[f"runs/{run}"] = {"launched_by": who, "outputs": f"gs://{BUCKET}/{prefix}",
                                    "sim": {"scenario": "scenarios/test.py:Test"}}
        for index, status in enumerate(statuses):
            self.docs[f"runs/{run}/episodes/{index}"] = {"index": index, "status": status}
            if status == "succeeded":
                self.add_episode_files(prefix, index, **episode_options)

    def add_episode_files(self, prefix: str, index: int, **options) -> None:
        folder = self.root / prefix / f"episode-{index}"
        episode_folder(folder, **options)
        for path in folder.rglob("*"):
            if path.is_file():
                self.files[f"{prefix}episode-{index}/{path.relative_to(folder).as_posix()}"] = path

    def finish_episode(self, run: str, index: int) -> None:
        self.docs[f"runs/{run}/episodes/{index}"]["status"] = "succeeded"
        self.add_episode_files(self.docs[f"runs/{run}"]["outputs"].split("/", 3)[3], index)

    def document(self, path):
        return self.docs.get(path)

    def documents(self, collection):
        return {p.rsplit("/", 1)[1]: d for p, d in self.docs.items() if p.rsplit("/", 1)[0] == collection}

    def objects(self, prefix):
        return [(name, path.stat().st_size) for name, path in sorted(self.files.items()) if name.startswith(prefix)]

    def download(self, name, path):
        self.downloads.append(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.files[name], path)


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.dict(os.environ, {"WILDHAT_CACHE": str(self.tmp / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cloud = FakeCloud(self.tmp / "bucket")
        self.model = self.tmp / "training" / "fuel_detector"
        self.model.mkdir(parents=True)
        self.said: list[str] = []

    def write(self, text: str) -> Path:
        path = self.model / "dataset.toml"
        path.write_text(text)
        return path

    def say(self, line: str) -> None:
        self.said.append(line)


class TestLoading(DatasetTestCase):
    def test_a_dataset(self):
        self.write('runs = ["20261102-0019-w22f"]\nevery = 2\nvalidation = 0.2\ncameras = ["front"]\n'
                   '[test]\nruns = ["20261106-0900-p7tz"]\n')
        ds = Dataset.load(self.model)  # the directory finds its dataset.toml
        self.assertEqual(ds.name, "fuel_detector")
        self.assertEqual((ds.runs, ds.test_runs, ds.every, ds.validation, ds.cameras),
                         (["20261102-0019-w22f"], ["20261106-0900-p7tz"], 2, 0.2, ["front"]))

    def test_defaults(self):
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f"]\n'))
        self.assertEqual((ds.test_runs, ds.every, ds.validation, ds.cameras), ([], 1, 0.15, None))

    def test_mistakes(self):
        for text, message in [
            ('run = ["20261102-0019-w22f"]\n', "unknown setting run"),
            ('every = 2\n', "needs runs"),
            ('runs = []\n', "runs is empty"),
            ('runs = ["fuel-1"]\n', "'fuel-1' in runs isn't a run ID"),
            ('runs = "20261102-0019-w22f"\n', "must be a list of run IDs"),
            ('runs = ["20261102-0019-w22f", "20261102-0019-w22f"]\n', "listed twice"),
            ('runs = ["20261102-0019-w22f"]\n[test]\nruns = ["20261102-0019-w22f"]\n', "in both runs and [test]"),
            ('runs = ["20261102-0019-w22f"]\n[test]\nepisodes = 3\n', "[test] takes only runs"),
            ('runs = ["20261102-0019-w22f"]\nevery = 0\n', "every must be"),
            ('runs = ["20261102-0019-w22f"]\nvalidation = 1\n', "validation must be"),
            ('runs = ["20261102-0019-w22f"]\ncameras = "front"\n', "cameras must be"),
            ('runs = [\n', "isn't valid TOML"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message.replace("[", r"\[")):
                Dataset.load(self.write(text))
        with self.assertRaisesRegex(ValueError, "There's no"):
            Dataset.load(self.tmp / "nowhere")


class TestSplit(DatasetTestCase):
    def test_the_split_is_by_episode_and_never_changes(self):
        self.assertEqual(split_share("20261102-0019-w22f", 3), split_share("20261102-0019-w22f", 3))
        self.assertNotEqual(split_share("20261102-0019-w22f", 3), split_share("20261102-0019-w22f", 4))
        one = Dataset.load(self.write('runs = ["20261102-0019-w22f"]\n'))
        two = Dataset.load(self.write('runs = ["20261102-0019-w22f", "20261104-1830-k3rq"]\n'))
        for i in range(100):  # adding a run moves nothing
            self.assertEqual(one.split_of("20261102-0019-w22f", i), two.split_of("20261102-0019-w22f", i))

    def test_about_the_right_share_goes_to_validation(self):
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f"]\nvalidation = 0.15\n'))
        counts = Counter(ds.split_of(f"20261102-{i // 100:04d}-abcd", i % 100) for i in range(2000))
        self.assertAlmostEqual(counts["validation"] / 2000, 0.15, delta=0.03)

    def test_test_runs_are_test_whole(self):
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f"]\n[test]\nruns = ["20261106-0900-p7tz"]\n'))
        self.assertEqual({ds.split_of("20261106-0900-p7tz", i) for i in range(50)}, {"test"})


class TestFetching(DatasetTestCase):
    RUN, OTHER = "20261102-0019-w22f", "20261106-0900-p7tz"

    def test_fetches_succeeded_episodes_without_previews(self):
        self.cloud.add_run(self.RUN, ["succeeded", "failed", "succeeded"])
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\n')).fetch(self.cloud, self.say)

        self.assertEqual(sorted(len(ds.episodes(s)) for s in ("train", "validation")), [0, 2])
        cached = self.tmp / "cache" / "runs" / self.RUN
        self.assertTrue((cached / "episode-0" / "episode.json").exists())
        self.assertFalse((cached / "episode-1").exists())
        self.assertFalse(any("preview" in name for name in self.cloud.downloads))
        self.assertIn(f"{self.RUN}: 1 episode(s) failed or stopped, so they're left out", self.said)

    def test_a_finished_run_is_fetched_once(self):
        self.cloud.add_run(self.RUN, ["succeeded"])
        Dataset.load(self.write(f'runs = ["{self.RUN}"]\n')).fetch(self.cloud, self.say)
        count = len(self.cloud.downloads)
        self.cloud.docs.clear()  # no records, no network needed
        ds = Dataset.load(self.model).fetch(self.cloud, self.say)
        self.assertEqual(len(self.cloud.downloads), count)
        self.assertEqual(len(ds.episodes("train") + ds.episodes("validation")), 1)

    def test_a_run_still_going_is_fetched_as_it_finishes(self):
        self.cloud.add_run(self.RUN, ["succeeded", "running"])
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = 0\n')).fetch(self.cloud, self.say)
        self.assertEqual(len(ds.episodes("train")), 1)
        self.assertIn(f"{self.RUN}: still going (1 episode(s) not done); fetch again later for the rest", self.said)

        self.cloud.finish_episode(self.RUN, 1)
        count = len(self.cloud.downloads)
        ds = Dataset.load(self.model).fetch(self.cloud, self.say)
        self.assertEqual(len(ds.episodes("train")), 2)
        self.assertTrue(all("episode-1/" in name for name in self.cloud.downloads[count:]))  # only the new one

    def test_episodes_without_an_episode_json_are_left_out(self):
        self.cloud.add_run(self.RUN, ["succeeded"])
        self.cloud.docs[f"runs/{self.RUN}/episodes/1"] = {"index": 1, "status": "succeeded"}
        self.cloud.files[f"runs/kai-morales/{self.RUN}/episode-1/results.csv"] = self.tmp / "results.csv"
        (self.tmp / "results.csv").write_text("a,b\n")
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = 0\n')).fetch(self.cloud, self.say)
        self.assertEqual(len(ds.episodes("train")), 1)
        self.assertIn(f"{self.RUN} episode 1 has no episode.json (not a simulation?), so it's left out", self.said)

    def test_a_fetch_that_stopped_partway_carries_on(self):
        self.cloud.add_run(self.RUN, ["succeeded"])
        name = f"runs/kai-morales/{self.RUN}/episode-0/episode.json"
        done = self.tmp / "cache" / "runs" / self.RUN / "episode-0" / "episode.json"
        done.parent.mkdir(parents=True)
        shutil.copyfile(self.cloud.files[name], done)
        Dataset.load(self.write(f'runs = ["{self.RUN}"]\n')).fetch(self.cloud, self.say)
        self.assertNotIn(name, self.cloud.downloads)
        self.assertTrue(self.cloud.downloads)

    def test_runs_that_arent_simulations(self):
        self.cloud.add_run(self.RUN, ["succeeded"])
        del self.cloud.docs[f"runs/{self.RUN}"]["sim"]
        with self.assertRaisesRegex(CloudError, "isn't a simulation run.*Take it out"):
            Dataset.load(self.write(f'runs = ["{self.RUN}"]\n')).fetch(self.cloud, self.say)

    def test_a_missing_run(self):
        with self.assertRaisesRegex(CloudError, f"There's no run {self.RUN}"):
            Dataset.load(self.write(f'runs = ["{self.RUN}"]\n')).fetch(self.cloud, self.say)

    def test_using_it_before_fetching(self):
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\n'))
        with self.assertRaisesRegex(CloudError, "isn't downloaded yet. Call fetch"):
            ds.samples("train")


class TestSamples(DatasetTestCase):
    RUN, OTHER = "20261102-0019-w22f", "20261106-0900-p7tz"

    def test_every_nth_frame_of_the_chosen_cameras(self):
        self.cloud.add_run(self.RUN, ["succeeded"], frames=5, cameras=("front", "back"))
        self.cloud.add_run(self.OTHER, ["succeeded"], frames=5, cameras=("front", "back"))
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nevery = 2\nvalidation = 0\ncameras = ["front"]\n'
                                     f'[test]\nruns = ["{self.OTHER}"]\n')).fetch(self.cloud, self.say)

        train = ds.samples("train")
        self.assertEqual([(s.run, s.index, s.camera, s.frame) for s in train],
                         [(self.RUN, 0, "front", 0), (self.RUN, 0, "front", 2), (self.RUN, 0, "front", 4)])
        self.assertEqual(train[1].image().shape, (24, 32, 3))
        self.assertEqual(train[1].mask().shape, (24, 32))
        self.assertEqual([b.label for b in train[1].boxes()], ["ball"])
        self.assertEqual(len(ds.samples("test")), 3)
        self.assertEqual(ds.samples("validation"), [])

        self.assertEqual(ds.summary(), {
            "train": {"episodes": 1, "samples": 3, "labels": {"ball": 3}},
            "validation": {"episodes": 0, "samples": 0, "labels": {}},
            "test": {"episodes": 1, "samples": 3, "labels": {"ball": 3}},
        })

    def test_a_camera_an_episode_doesnt_have(self):
        self.cloud.add_run(self.RUN, ["succeeded"])
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = 0\ncameras = ["side"]\n'))
        with self.assertRaisesRegex(KeyError, "has no camera side \\(it has front\\)"):
            ds.fetch(self.cloud, self.say).samples("train")

    def test_no_such_split(self):
        with self.assertRaisesRegex(ValueError, "no split 'val'"):
            Dataset.load(self.write(f'runs = ["{self.RUN}"]\n')).samples("val")


class TestCheck(DatasetTestCase):
    def test_reports_each_run_and_the_split(self):
        self.cloud.add_run("20261102-0019-w22f", ["succeeded", "succeeded", "failed"])
        self.cloud.add_run("20261104-1830-k3rq", ["succeeded", "running"], who="dana-wells")
        self.cloud.add_run("20261105-1200-zzzz", ["failed"])
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f", "20261104-1830-k3rq", "20261105-1200-zzzz", '
                                     '"20261107-0000-nope"]\nvalidation = 0\n'))

        self.assertFalse(ds.check(self.cloud, self.say))
        self.assertEqual(self.said[0], "! 20261102-0019-w22f: 2 episode(s) succeeded (1 failed), by kai-morales")
        self.assertEqual(self.said[1], "! 20261104-1830-k3rq: 1 episode(s) succeeded (1 running), by dana-wells")
        self.assertIn("still going", self.said[2])
        self.assertEqual(self.said[3], "✗ 20261105-1200-zzzz: no episode succeeded (1 failed)")
        self.assertEqual(self.said[4], "✗ 20261107-0000-nope: there's no such run")
        self.assertEqual(self.said[5], "Episodes: train 3, validation 0, test 0")
        self.assertFalse(self.cloud.downloads)

    def test_training_jobs_and_other_jobs(self):
        self.cloud.add_run("20261102-0019-w22f", ["succeeded"])
        self.cloud.add_run("20261104-1830-k3rq", ["succeeded"])
        for run in ("20261102-0019-w22f", "20261104-1830-k3rq"):
            del self.cloud.docs[f"runs/{run}"]["sim"]
        self.cloud.docs["runs/20261102-0019-w22f"]["training"] = "20261104-0738-ekkd"
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f", "20261104-1830-k3rq"]\n'))
        self.assertFalse(ds.check(self.cloud, self.say))
        self.assertEqual(self.said[:2], [
            "✗ 20261102-0019-w22f: it's a training job (training run 20261104-0738-ekkd), not a simulation",
            "✗ 20261104-1830-k3rq: it isn't a simulation run, so it has nothing to train on"])

    def test_few_validation_episodes(self):
        self.cloud.add_run("20261102-0019-w22f", ["succeeded"] * 20)
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f"]\nvalidation = 0.1\n'))
        ds.check(self.cloud, self.say)
        self.assertTrue(any(line.startswith("! Only ") and "validation episode(s)" in line for line in self.said))

    def test_all_good(self):
        self.cloud.add_run("20261102-0019-w22f", ["succeeded"] * 20)
        ds = Dataset.load(self.write('runs = ["20261102-0019-w22f"]\n'))
        self.assertTrue(ds.check(self.cloud, self.say))
        self.assertTrue(self.said[0].startswith("✓ 20261102-0019-w22f: 20 episode(s) succeeded, by kai-morales"))


class TestWarnings(DatasetTestCase):
    """Splits whose scores won't mean much, as fetch() warns."""

    RUN, OTHER = "20261102-0019-w22f", "20261106-0900-p7tz"

    def test_a_small_unlike_test_split(self):
        self.cloud.add_run(self.RUN, ["succeeded"] * 3, frames=40)  # 4 px balls
        self.cloud.add_run(self.OTHER, ["succeeded"], frames=4, box=(1, 1, 21, 13))  # 20 px across
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = 0\n[test]\nruns = ["{self.OTHER}"]\n'))
        ds.fetch(self.cloud, self.say)
        self.assertEqual(ds.warnings(), [
            "! test has only 1 episode(s), so its scores will be noisy. Runs with more, shorter episodes split better.",
            "! test has few boxes of ball 4, so its scores for them will be noisy.",
            "! test's boxes are a different size from train's (median across: ball 20 px against 4), "
            "so its scores say little about train's kind of picture.",
        ])
        self.assertEqual(self.said[-3:], ds.warnings())  # fetch said them

    def test_a_good_split_says_nothing(self):
        self.cloud.add_run(self.RUN, ["succeeded"] * 3, frames=40)
        self.cloud.add_run(self.OTHER, ["succeeded"] * 5, frames=40, box=(1, 1, 7, 7))
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = 0\n[test]\nruns = ["{self.OTHER}"]\n'))
        self.assertEqual(ds.fetch(self.cloud, self.say).warnings(), [])


class TestOldFormat(DatasetTestCase):
    """The first runs (format 1) have pictures but no tables."""

    RUN, NEW = "20261030-2208-rfsf", "20261102-0019-w22f"

    def setUp(self):
        super().setUp()
        self.cloud.add_run(self.RUN, ["succeeded", "succeeded"])
        for name in [n for n in self.cloud.files if n.startswith(f"runs/kai-morales/{self.RUN}/") and
                     n.endswith(".parquet")]:
            del self.cloud.files[name]
        self.cloud.add_run(self.NEW, ["succeeded"])

    def test_check_says_so(self):
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}", "{self.NEW}"]\nvalidation = 0\n'))
        self.assertFalse(ds.check(self.cloud, self.say))
        self.assertEqual(self.said[0], f"✗ {self.RUN}: it's in the old format, from before the episode tables, "
                                       "so it can't be trained on")
        self.assertTrue(self.said[1].startswith(f"✓ {self.NEW}"))

    def test_fetch_leaves_them_out(self):
        ds = Dataset.load(self.write(f'runs = ["{self.RUN}", "{self.NEW}"]\nvalidation = 0\n'))
        ds.fetch(self.cloud, self.say)
        self.assertIn(f"{self.RUN}: 2 episode(s) are in the old format, from before the episode tables, "
                      "so they're left out", self.said)
        self.assertEqual([s.run for s in ds.samples("train")], [self.NEW] * 4)
        self.assertFalse(any(self.RUN in name for name in self.cloud.downloads))


class TestYolo(DatasetTestCase):
    RUN, OTHER = "20261102-0019-w22f", "20261106-0900-p7tz"

    def dataset(self, validation=0.5, full=None) -> Dataset:
        # Enough episodes that both train and validation get some.
        self.cloud.add_run(self.RUN, ["succeeded"] * 6, frames=3, full=full)
        self.cloud.add_run(self.OTHER, ["succeeded"], frames=3, full=full)
        return Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = {validation}\n'
                                       f'[test]\nruns = ["{self.OTHER}"]\n')).fetch(self.cloud, self.say)

    def test_exports_pictures_labels_and_data_yaml(self):
        ds = self.dataset()
        data_yaml = ds.yolo(["ball"], say=self.say)
        folder = data_yaml.parent
        self.assertEqual(folder.parent, self.tmp / "cache" / "yolo")
        self.assertTrue(folder.name.startswith("fuel_detector-"))
        self.assertEqual(data_yaml.read_text().splitlines()[1:], [
            f'path: "{folder}"', "train: images/train", "val: images/validation", "test: images/test",
            "names:", '  0: "ball"'])

        for split in ("train", "validation", "test"):
            samples = ds.samples(split)
            labels = sorted((folder / "labels" / split).iterdir())
            self.assertEqual(len(labels), len(samples))
            self.assertEqual(len(list((folder / "images" / split).iterdir())), len(samples))
        s = ds.samples("test")[1]
        stem = f"{self.OTHER}_0_front_000001"
        # The box is x 1-5, y 1-5 in a 32x24 picture.
        self.assertEqual((folder / "labels" / "test" / f"{stem}.txt").read_text(),
                         "0 0.093750 0.125000 0.125000 0.166667\n")
        picture = folder / "images" / "test" / f"{stem}.jpg"
        self.assertEqual(picture.stat().st_ino, (s.episode.folder / "cam" / "front" / "000001.jpg").stat().st_ino)
        self.assertIn(f"YOLO dataset: {folder}", self.said)
        self.assertIn("test 3 picture(s), 3 box(es)", self.said[-1])

    def test_the_same_export_is_reused(self):
        ds = self.dataset()
        first = ds.yolo(["ball"], say=self.say)
        (first.parent / "marker").touch()
        self.assertEqual(ds.yolo(["ball"], say=self.say), first)
        self.assertTrue((first.parent / "marker").exists())
        self.assertIn("(already exported)", self.said[-1])
        self.assertNotEqual(ds.yolo(["ball"], min_pixels=20, say=self.say), first)

    def test_other_labels_and_hidden_objects_leave_empty_label_files(self):
        ds = self.dataset()
        for options in [dict(classes=["fuel"]), dict(classes=["ball"], min_pixels=20)]:  # the box covers 16
            folder = ds.yolo(say=self.say, **options).parent
            texts = {p.read_text() for p in (folder / "labels" / "train").iterdir()}
            self.assertEqual(texts, {""})
        self.assertIn("  ! No training boxes labeled fuel. Is that the right label?", self.said)

    def test_full_boxes(self):
        # Seen: x 1-5, y 1-5 (16 pixels). Whole: x 1-9, y 1-5 (32 pixels), so half seen.
        ds = self.dataset(full=(1, 1, 9, 5, 32))
        stem = f"{self.OTHER}_0_front_000001"
        visible = ds.yolo(["ball"], say=self.say).parent
        full = ds.yolo(["ball"], boxes="full", say=self.say).parent
        self.assertNotEqual(full, visible)
        self.assertEqual((full / "labels" / "test" / f"{stem}.txt").read_text(),
                         "0 0.156250 0.125000 0.250000 0.166667\n")
        self.assertEqual((visible / "labels" / "test" / f"{stem}.txt").read_text(),
                         "0 0.093750 0.125000 0.125000 0.166667\n")
        # Half seen: kept at 0.5, left out above it.
        for share, expected in [(0.5, "0 0.093750"), (0.6, "")]:
            folder = ds.yolo(["ball"], min_share=share, say=self.say).parent
            self.assertTrue((folder / "labels" / "test" / f"{stem}.txt").read_text().startswith(expected))

    def test_full_boxes_need_new_episodes(self):
        ds = self.dataset()
        for options in [dict(boxes="full"), dict(min_share=0.5)]:
            with self.subTest(**options), self.assertRaisesRegex(ValueError, "simulated before they were added"):
                ds.yolo(["ball"], say=self.say, **options)
        self.assertEqual(list((self.tmp / "cache" / "yolo").glob("*.building-*")), [])

    def test_two_classes(self):
        folder = self.dataset().yolo(["fuel", "ball"], say=self.say).parent
        self.assertIn('  0: "fuel"\n  1: "ball"\n', (folder / "data.yaml").read_text())
        self.assertTrue(next((folder / "labels" / "train").iterdir()).read_text().startswith("1 "))

    def test_boxes_are_clipped_to_the_picture(self):
        self.assertEqual(_yolo_line(0, {"x0": -4, "y0": 20, "x1": 4, "y1": 30}, 32, 24),
                         "0 0.062500 0.916667 0.125000 0.166667")
        self.assertIsNone(_yolo_line(0, {"x0": 40, "y0": 0, "x1": 50, "y1": 4}, 32, 24))

    def test_mistakes(self):
        ds = self.dataset()
        for kwargs, message in [(dict(classes=[]), "classes must be"), (dict(classes="ball"), "classes must be"),
                                (dict(classes=["ball", "ball"]), "in classes twice"),
                                (dict(classes=["ball"], min_pixels=-1), "min_pixels"),
                                (dict(classes=["ball"], boxes="whole"), "boxes must be"),
                                (dict(classes=["ball"], min_share=2), "min_share")]:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                ds.yolo(say=self.say, **kwargs)
        no_validation = Dataset.load(self.write(f'runs = ["{self.RUN}"]\nvalidation = 0\n'))
        with self.assertRaisesRegex(ValueError, "no validation episodes, and YOLO needs some"):
            no_validation.yolo(["ball"], say=self.say)


class TestCommandLine(DatasetTestCase):
    def test_fetch_prints_the_counts(self):
        self.cloud.add_run("20261102-0019-w22f", ["succeeded"], frames=2)
        self.write('runs = ["20261102-0019-w22f"]\nvalidation = 0\n')
        with mock.patch.object(datasets, "Cloud", lambda: self.cloud), \
                mock.patch("builtins.print") as printed:
            self.assertEqual(datasets.main(["fetch", str(self.model)]), 0)
        lines = [c.args[0] for c in printed.call_args_list]
        self.assertIn("train: 1 episode(s), 2 sample(s); ball 2", lines)
        self.assertIn("validation: 0 episode(s), 0 sample(s); no boxes", lines)

    def test_yolo(self):
        self.cloud.add_run("20261102-0019-w22f", ["succeeded"] * 6, frames=2)
        self.write('runs = ["20261102-0019-w22f"]\nvalidation = 0.5\n')
        with mock.patch.object(datasets, "Cloud", lambda: self.cloud), \
                mock.patch("builtins.print") as printed:
            self.assertEqual(datasets.main(["yolo", str(self.model), "--classes", "ball"]), 0)
        self.assertTrue(str(printed.call_args_list[-1].args[0]).endswith("data.yaml"))
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            datasets.main(["yolo", str(self.model)])

    def test_mistakes_are_one_line(self):
        self.write('runs = ["oops"]\n')
        with mock.patch("sys.stderr") as err:
            self.assertEqual(datasets.main(["check", str(self.model)]), 1)
        self.assertIn("isn't a run ID", "".join(c.args[0] for c in err.write.call_args_list))


if __name__ == "__main__":
    unittest.main()
