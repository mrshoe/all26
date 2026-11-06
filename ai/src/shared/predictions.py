"""A model's predictions on a dataset's pictures, written as episode tables
(the same `boxes` and `keypoints` schemas as the ground truth, with
`source` the model's name), for `shared.score` to compare with the truth.

    from shared.datasets import Dataset
    from shared.predictions import Predictions

    ds = Dataset.load("training/fuel_detector").fetch()
    preds = Predictions(run.folder / "predictions", model="fuel_detector", classes=["fuel"])
    for sample in ds.samples("validation"):
        for x0, y0, x1, y1, score in my_model(sample.image()):
            preds.box(sample, "fuel", x0, y0, x1, y1, score)
        preds.done(sample)            # pictures with no predictions count too
    preds.finish(ds, "validation")    # then: python -m shared.score <folder>

Coordinates are pixels in the picture, (0, 0) its top-left corner, as the
truth's are; a box's (x1, y1) is one past its bottom-right corner.

Folder:

    predictions.json                      which model, dataset and split, and
                                          which pictures it was run on
    <run>/episode-<n>/boxes.parquet       its boxes, in the episode's pictures
    <run>/episode-<n>/keypoints.parquet   its keypoints
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from shared.logger import SCHEMAS, plain


class Predictions:
    def __init__(self, folder, model: str, classes: Optional[list[str]] = None):
        """`model` names the predictions' source, e.g. "fuel_detector" or
        "fuel_detector@20261104-0738-ekkd". `classes` are the labels (and
        keypoint classes) it was trained to find: those are what it's scored
        on. Left out, they're the ones it predicted at least once, so a class
        it never finds isn't scored at all; say them."""
        if not model or model == "truth":
            raise ValueError('Name the model: Predictions(folder, model="fuel_detector")')
        if classes is not None and (isinstance(classes, str) or not all(isinstance(c, str) and c for c in classes)):
            raise ValueError('classes must be a list of labels, like ["fuel"]')
        self.folder = Path(folder)
        self.model = model
        self.classes = list(classes) if classes is not None else None
        self.rows: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self.pictures: set[tuple] = set()

    def _where(self, sample) -> tuple[tuple, int, int]:
        """The sample's episode (run, index), camera id and frame."""
        return (sample.run, sample.index), sample.episode.objects[sample.camera]["object"], sample.frame

    def box(self, sample, label: str, x0: float, y0: float, x1: float, y1: float, score: float) -> None:
        """A box the model found in this sample's picture, with its confidence (0 to 1)."""
        values = (x0, y0, x1, y1, score)
        if not all(math.isfinite(_number(v)) for v in values):
            raise ValueError(f"A box needs numbers: {values}")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"A box's x1 and y1 must be past its x0 and y0: {values[:4]}")
        episode, camera, frame = self._where(sample)
        self.rows[episode]["boxes"].append({
            "frame": frame, "camera": camera, "label": str(label),
            "x0": _number(x0), "y0": _number(y0), "x1": _number(x1), "y1": _number(y1),
            "score": _number(score), "source": self.model})
        self.pictures.add((*episode, camera, frame))

    def keypoint(self, sample, label: str, x: float, y: float, score: float, index: int = 0,
                 visible: Optional[bool] = None) -> None:
        """A keypoint the model found: its class, pixel and confidence;
        `index` if the model tells which of an object's points it is, and
        `visible` if it says whether the point is seen."""
        if not all(math.isfinite(_number(v)) for v in (x, y, score)):
            raise ValueError(f"A keypoint needs numbers: {(x, y, score)}")
        episode, camera, frame = self._where(sample)
        width, height = _size(sample)
        self.rows[episode]["keypoints"].append({
            "frame": frame, "camera": camera, "label": str(label), "index": int(index),
            "x": _number(x), "y": _number(y), "in_frame": bool(0 <= x < width and 0 <= y < height),
            "visible": visible, "score": _number(score), "source": self.model})
        self.pictures.add((*episode, camera, frame))

    def done(self, sample) -> None:
        """Say the model looked at this picture, even if it found nothing:
        then the scorer counts what it missed there."""
        episode, camera, frame = self._where(sample)
        self.pictures.add((*episode, camera, frame))

    def finish(self, dataset=None, split: Optional[str] = None) -> Path:
        """Write the tables and predictions.json; returns the folder."""
        self.folder.mkdir(parents=True, exist_ok=True)
        episodes = sorted({p[:2] for p in self.pictures})
        for run, index in episodes:
            where = self.folder / run / f"episode-{index}"
            where.mkdir(parents=True, exist_ok=True)
            for table in ("boxes", "keypoints"):
                rows = self.rows[(run, index)][table]
                path = where / f"{table}.parquet"
                if not rows:
                    path.unlink(missing_ok=True)
                    continue
                schema = SCHEMAS[table]
                rows = sorted(rows, key=lambda r: (r["frame"], r["camera"], -r["score"]))
                pq.write_table(pa.Table.from_pylist([{c: plain(r.get(c)) for c in schema.names} for r in rows],
                                                    schema=schema), path, compression="zstd")
        info = {
            "model": self.model,
            "dataset": str(dataset.path) if dataset is not None else None,
            "split": split,
            "classes": self.classes if self.classes is not None else sorted(
                {r["label"] for t in self.rows.values() for rows in t.values() for r in rows}),
            "pictures": [list(p) for p in sorted(self.pictures)],
            "boxes": sum(len(t["boxes"]) for t in self.rows.values()),
            "keypoints": sum(len(t["keypoints"]) for t in self.rows.values()),
        }
        (self.folder / "predictions.json").write_text(json.dumps(info) + "\n")
        return self.folder


def _number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _size(sample) -> tuple[int, int]:
    cameras = sample.episode.table("cameras").to_pylist()
    camera = sample.episode.objects[sample.camera]["object"]
    row = next(r for r in cameras if r["object"] == camera)
    return row["width"], row["height"]


def load(folder) -> tuple[dict, dict]:
    """A predictions folder: its predictions.json, and {(run, index):
    {"boxes": [...], "keypoints": [...]}} as rows."""
    folder = Path(folder)
    info_path = folder / "predictions.json"
    if not info_path.exists():
        raise ValueError(f"{folder} has no predictions.json: write it with Predictions(...).finish()")
    info = json.loads(info_path.read_text())
    rows: dict = defaultdict(lambda: {"boxes": [], "keypoints": []})
    for run, index in {tuple(p[:2]) for p in info["pictures"]}:
        for table in ("boxes", "keypoints"):
            path = folder / run / f"episode-{index}" / f"{table}.parquet"
            if path.exists():
                rows[(run, index)][table] = pq.read_table(path).to_pylist()
    return info, rows
