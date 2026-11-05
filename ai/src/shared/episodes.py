"""Reading an episode's folder, as `logger.Logger` writes it (format 2),
e.g. after `wildhat download`:

    ep = Episode("runs/20261030-2130-abcd/episode-0")
    ep.cameras                  ["front"]
    ep.objects["ball"]          {"object": 3, "type": "Sphere", "label": "ball", ...}
    ep.image("front", 12)       height x width x 3 uint8
    ep.mask("front", 12)        height x width uint16 object ids (simulation)
    ep.boxes("front", 12)       [Box(label="ball", name="ball", x0=..., ...)]
    ep.keypoints("front", 12)   [Keypoint(label="corner", name="crate", index=0, x=..., visible=True, ...)]
    ep.table("poses")           a pyarrow Table; .to_pandas() if you have pandas

Needs numpy, pillow and pyarrow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image


@dataclass
class Box:
    """A box in one picture, in pixels; (x1, y1) is one past the bottom-right
    corner. `name` is the object's (None for a model's detection), `pixels`
    how many it covers (ground truth only), `score` the model's confidence
    (None for ground truth)."""

    label: str
    name: Optional[str]
    x0: float
    y0: float
    x1: float
    y1: float
    pixels: Optional[int] = None
    score: Optional[float] = None
    source: str = "truth"
    # Simulation only: the box of the object's whole shape, as if nothing
    # hid it (x0, y0, x1, y1, cut to the picture), and how many pixels it
    # would cover.
    full: Optional[tuple[float, float, float, float]] = None
    full_pixels: Optional[int] = None

    @property
    def share(self) -> Optional[float]:
        """How much of the object is seen, 0 to 1: pixels / full_pixels."""
        if not self.pixels or not self.full_pixels:
            return None
        return min(1.0, self.pixels / self.full_pixels)


@dataclass
class Keypoint:
    """A named point on an object, in one picture. `label` is its class
    ("corner") and `index` which of the object's points of that class. `x`
    and `y` are pixels (it may be outside the picture: see `in_frame`);
    `depth` is meters in front of the camera. `occluded_by` is None if it's
    seen or out of the picture, "self" if its own object's far side hides
    it, else the name of what's in front."""

    label: str
    name: Optional[str]
    index: int
    x: float
    y: float
    depth: Optional[float]
    in_frame: bool
    visible: Optional[bool]
    occluded_by: Optional[str] = None
    score: Optional[float] = None
    source: str = "truth"


class Episode:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "episode.json").read_text())
        if self.info.get("format", 1) < 2:
            raise ValueError(f"{self.folder} is in an old format (1), from before the episode tables; "
                             "only its videos can still be viewed.")

    @property
    def frames(self) -> int:
        return self.info["frames"]

    @cache
    def table(self, name: str) -> pa.Table:
        return pq.read_table(self.folder / f"{name}.parquet")

    @cached_property
    def objects(self) -> dict[str, dict]:
        """Every object by name, with its settings decoded."""
        rows = self.table("objects").to_pylist()
        return {r["name"]: {**r, "settings": json.loads(r["settings"])} for r in rows}

    @cached_property
    def names(self) -> dict[int, str]:
        """Object ids to names."""
        return {o["object"]: name for name, o in self.objects.items()}

    @property
    def cameras(self) -> list[str]:
        if "cameras" not in self.info["tables"]:
            return []
        return [self.names[i] for i in self.table("cameras")["object"].to_pylist()]

    def _image_row(self, camera: str, frame: int) -> dict:
        images = self.table("images")
        found = images.filter(pc.and_(pc.equal(images["camera"], self.objects[camera]["object"]),
                                      pc.equal(images["frame"], frame))).to_pylist()
        if not found:
            raise KeyError(f"{camera} has no picture in frame {frame}")
        return found[0]

    def image(self, camera: str, frame: int) -> np.ndarray:
        with Image.open(self.folder / self._image_row(camera, frame)["image"]) as img:
            return np.asarray(img.convert("RGB"))

    def mask(self, camera: str, frame: int) -> np.ndarray:
        path = self._image_row(camera, frame)["mask"]
        if path is None:
            raise KeyError(f"{camera} has no mask in frame {frame}")
        with Image.open(self.folder / path) as img:
            return np.asarray(img).astype(np.uint16)

    def _rows(self, table: str, camera: str, frame: int, source: str) -> list[dict]:
        if table not in self.info["tables"]:
            return []
        t = self.table(table)
        return t.filter(pc.and_(pc.and_(pc.equal(t["camera"], self.objects[camera]["object"]),
                                        pc.equal(t["frame"], frame)),
                                pc.equal(t["source"], source))).to_pylist()

    def boxes(self, camera: str, frame: int, source: str = "truth") -> list[Box]:
        """The boxes in this camera's picture in this frame, from `source`:
        "truth", or a model's name."""
        out = []
        for r in self._rows("boxes", camera, frame, source):
            full = None
            if r.get("full_x0") is not None:
                full = (r["full_x0"], r["full_y0"], r["full_x1"], r["full_y1"])
            out.append(Box(r["label"], self.names.get(r["object"]), r["x0"], r["y0"], r["x1"], r["y1"],
                           r["pixels"], r["score"], r["source"], full, r.get("full_pixels")))
        return out

    def keypoints(self, camera: str, frame: int, source: str = "truth") -> list[Keypoint]:
        """Every keypoint in this camera's picture in this frame, seen or
        hidden, from `source`: "truth", or a model's name. (Episodes from
        before keypoints have none.)"""
        out = []
        for r in self._rows("keypoints", camera, frame, source):
            by = r["occluded_by"]
            by = None if by is None else "self" if by == r["object"] else self.names.get(by)
            out.append(Keypoint(r["label"], self.names.get(r["object"]), r["index"], r["x"], r["y"],
                                r["depth"], r["in_frame"], r["visible"], by, r["score"], r["source"]))
        return out
