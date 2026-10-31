"""Reading an episode's folder, as `logger.Logger` writes it, e.g. after
`wildhat download`:

    ep = Episode("runs/20261030-2130-abcd/episode-0")
    ep.cameras                  ["front"]
    ep.image("front", 12)       height x width x 3 uint8
    ep.mask("front", 12)        height x width uint16 object ids (simulation)
    ep.boxes("front", 12)       [Box(label="ball", name="ball", x0=..., ...)]
    ep.signals["ball.z"]        one value per frame

Needs numpy, pillow and pyarrow.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


@dataclass
class Box:
    """A labeled object's box in one picture, in pixels; (x1, y1) is one past
    the bottom-right corner."""

    label: str
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    pixels: int


class Episode:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "episode.json").read_text())

    @property
    def cameras(self) -> list[str]:
        return list(self.info["cameras"])

    @property
    def frames(self) -> int:
        return self.info["frames"]

    @cached_property
    def signals(self) -> dict[str, list]:
        return pq.read_table(self.folder / "signals.parquet").to_pydict()

    def image(self, camera: str, frame: int) -> np.ndarray:
        with Image.open(self.folder / "cam" / camera / f"{frame:06d}.jpg") as img:
            return np.asarray(img.convert("RGB"))

    def mask(self, camera: str, frame: int) -> np.ndarray:
        with Image.open(self.folder / "mask" / camera / f"{frame:06d}.png") as img:
            return np.asarray(img).astype(np.uint16)

    def boxes(self, camera: str, frame: int) -> list[Box]:
        """The labeled objects this camera sees in this frame."""
        out = []
        for name, label in self.info["labels"].items():
            key = f"{camera}.{name}"
            if f"{key}.x0" not in self.signals:
                continue
            x0, y0, x1, y1, n = (self.signals[f"{key}.{c}"][frame] for c in ("x0", "y0", "x1", "y1", "pixels"))
            if n and not any(v is None or math.isnan(v) for v in (x0, y0, x1, y1)):
                out.append(Box(label, name, x0, y0, x1, y1, int(n)))
        return out
