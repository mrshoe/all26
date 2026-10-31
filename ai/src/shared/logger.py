"""`Logger`: writes an episode's data to a folder, frame by frame. The
simulator and the robot both use it, and `episodes.Episode` reads it back.

    episode.json             what ran (whatever the caller says), plus the
                             frame count and signal names
    cam/<camera>/000000.jpg  what each camera saw, one file per frame
    mask/<camera>/000000.png which object each pixel shows, 16-bit ids
                             (simulation only; real cameras have none)
    signals.parquet          one row per frame: `frame`, `time`, then any
                             numbers the caller logs

Signal columns are named <thing>.<what>, e.g. ball.z or front.qw. A signal
missing from a frame is NaN there, so signals can come and go.

The caller's episode.json fields that `PostProcessor` and the visualizer
read: `fps`, `cameras` (camera name to its intrinsics) and `labels`
(object name to its class), with boxes logged as
<camera>.<object>.x0/y0/x1/y1/pixels in pixels, (x1, y1) one past the
bottom-right corner.

Streaming to the visualizer will be a second place frames go, next to disk.

Needs numpy, pillow and pyarrow.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

FORMAT = 1
JPEG_QUALITY = 90


class Logger:
    def __init__(self, folder, info: Optional[dict] = None):
        """`info` goes in episode.json; `finish` can add more."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.info = dict(info or {})
        self.columns: dict[str, list] = {"frame": [], "time": []}

    @property
    def frames(self) -> int:
        return len(self.columns["frame"])

    def frame(self, index: int, t: float, signals: Mapping[str, float] = {},
              images: Mapping[str, np.ndarray] = {}, masks: Mapping[str, np.ndarray] = {}) -> None:
        """Log one frame: numbers by signal name, and per camera an RGB image
        (height x width x 3 uint8) and, optionally, an object-id mask
        (height x width integers)."""
        n = self.frames
        row = {"frame": index, "time": t, **signals}
        for key, value in row.items():
            column = self.columns.setdefault(key, [math.nan] * n)  # new: NaN before
            column.append(value)
        for column in self.columns.values():
            if len(column) == n:  # not in this frame
                column.append(math.nan)
        stem = f"{index:06d}"
        for camera, rgb in images.items():
            path = self.folder / "cam" / camera / f"{stem}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path, quality=JPEG_QUALITY)
        for camera, ids in masks.items():
            path = self.folder / "mask" / camera / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(ids).astype(np.uint16)).save(path)

    def finish(self, info: Optional[dict] = None) -> dict:
        """Write signals.parquet and episode.json; return episode.json's
        contents."""
        table = pa.table({k: pa.array(v, type=pa.int32() if k == "frame" else pa.float64())
                          for k, v in self.columns.items()})
        pq.write_table(table, self.folder / "signals.parquet")
        episode = {"format": FORMAT, **self.info, **(info or {}),
                   "frames": self.frames, "signals": list(self.columns)}
        text = json.dumps(episode, indent=2, default=plain)
        (self.folder / "episode.json").write_text(text + "\n")
        return json.loads(text)


def plain(value):
    """numpy values as JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Can't write {type(value).__name__} to episode.json")
