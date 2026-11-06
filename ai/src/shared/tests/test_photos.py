import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import os

import numpy as np
from PIL import Image

from shared import photos
from shared.datasets import Dataset
from shared.episodes import Episode
from shared.predictions import Predictions
from shared.score import score

BUCKET = "test-bucket"


class FakeCloud:
    handle = "kai-morales"
    bucket = BUCKET

    def __init__(self, root: Path):
        self.root = root
        self.docs: dict[str, dict] = {}

    def upload(self, name, data, content_type="application/octet-stream"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data, path)

    def create(self, path, fields):
        assert path not in self.docs
        self.docs[path] = json.loads(json.dumps(fields, default=str))

    def document(self, path):
        return self.docs.get(path)

    def documents(self, collection):
        return {p.rsplit("/", 1)[1]: d for p, d in self.docs.items() if p.rsplit("/", 1)[0] == collection}

    def objects(self, prefix):
        return [(p.relative_to(self.root).as_posix(), p.stat().st_size)
                for p in sorted(self.root.rglob("*")) if p.is_file()
                and p.relative_to(self.root).as_posix().startswith(prefix)]

    def download(self, name, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / name, path)


def yolo_folder(root: Path) -> Path:
    """Two splits of 64 x 48 photos, as Roboflow exports them."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("train: ../train/images\nnc: 2\nnames: ['Bumper', 'Fuel']\n")
    for split, n in (("valid", 2), ("test", 1)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            Image.fromarray(np.full((48, 64, 3), 40 * i, np.uint8)).save(root / split / "images" / f"p{i}_jpg.rf.x.jpg")
            # a FUEL box from (16, 12) to (32, 24), and a bumper
            (root / split / "labels" / f"p{i}_jpg.rf.x.txt").write_text("1 0.375 0.375 0.25 0.25\n0 0.5 0.9 1.0 0.2\n")
    return root


class PhotosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.dict(os.environ, {"WILDHAT_CACHE": str(self.tmp / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cloud = FakeCloud(self.tmp / "bucket")
        self.folder = yolo_folder(self.tmp / "export")

    def test_class_names(self):
        self.assertEqual(photos.class_names(self.folder), ["Bumper", "Fuel"])
        (self.folder / "data.yaml").write_text("names:\n  0: Fuel\n  1: Bumper\n")
        self.assertEqual(photos.class_names(self.folder), ["Fuel", "Bumper"])
        (self.folder / "data.yaml").write_text("names:\n- Fuel\n- Bumper\n")
        self.assertEqual(photos.class_names(self.folder), ["Fuel", "Bumper"])

    def test_import_then_score(self):
        run = photos.import_photos(self.folder, ["valid", "test"], {"Fuel": "fuel"}, {"name": "test photos"},
                                   cloud=self.cloud, say=lambda *_: None)
        record = self.cloud.docs[f"runs/{run}"]
        self.assertEqual(record["photos"]["labels"], ["fuel"])
        episode = self.cloud.docs[f"runs/{run}/episodes/1"]
        self.assertEqual(episode["status"], "succeeded")
        self.assertEqual(episode["photos"]["previews"], ["preview/photo.mp4"])
        self.assertTrue((self.cloud.root / f"runs/kai-morales/{run}/episode-1/preview/photo.mp4").exists())

        model = self.tmp / "training" / "fuel_detector"
        model.mkdir(parents=True)
        (model / "dataset.toml").write_text(f'runs = ["20261101-0000-aaaa"]\n[test]\nruns = ["{run}"]\n')
        ds = Dataset.load(model)
        empty = self.tmp / "cache" / "runs" / "20261101-0000-aaaa" / "run.json"  # a train run, with nothing
        empty.parent.mkdir(parents=True)
        empty.write_text('{"episodes": [], "finished": true}')
        from shared.datasets import fetch_run
        fetch_run(self.cloud, run, lambda *_: None)
        samples = ds.samples("test")
        self.assertEqual(len(samples), 3)
        boxes = samples[0].boxes()
        self.assertEqual([(b.label, b.x0, b.y0, b.x1, b.y1, b.pixels) for b in boxes],
                         [("fuel", 16, 12, 32, 24, None)])  # the bumper isn't taken
        ep = Episode(samples[0].episode.folder)
        self.assertEqual(ep.table("photos")["file"].to_pylist(), ["p0_jpg.rf.x.jpg", "p1_jpg.rf.x.jpg"])
        self.assertEqual(samples[1].image()[0, 0].tolist(), [40, 40, 40])

        preds = Predictions(self.tmp / "preds", model="m", classes=["fuel"])
        for s in samples:
            preds.box(s, "fuel", 16, 12, 32, 24, 0.9)
            preds.done(s)
        preds.finish(ds, "test")
        result = score(self.tmp / "preds", dataset=ds)
        self.assertEqual(result["boxes"]["classes"]["fuel"]["ap50"], 1.0)
        # Photos have no pixel counts: min_pixels compares with the box's area (16 x 12 = 192).
        self.assertEqual(score(self.tmp / "preds", dataset=ds, min_pixels=192)["boxes"]["classes"]["fuel"]["truth"], 3)
        self.assertEqual(score(self.tmp / "preds", dataset=ds, min_pixels=193)["boxes"]["classes"]["fuel"]["truth"], 0)


if __name__ == "__main__":
    unittest.main()
