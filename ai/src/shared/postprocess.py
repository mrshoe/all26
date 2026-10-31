"""`PostProcessor`: after an episode, make the previews the visualizer plays,
from a folder `logger.Logger` wrote. The simulator runs it after each
episode, and the robot's upload tool before uploading.

    preview/<camera>.mp4        the camera's frames as a small H.264 video
    preview/<camera>.boxes.mp4  the same, with labeled objects' boxes drawn

Previews are low bitrate on purpose: the visualizer streams them out of the
bucket, and every GB downloaded costs (ai/INFRA.md). The full-quality
frames are still in cam/.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

BOX_COLOR = (255, 40, 200)
# Higher is smaller and blurrier; 23 is ffmpeg's default.
QUALITY = 30


class PostProcessor:
    def __init__(self, folder: Path, episode: dict):
        self.folder, self.episode = Path(folder), episode

    def run(self) -> list[str]:
        """Make every camera's previews; return their paths in the folder."""
        made = []
        signals = pq.read_table(self.folder / "signals.parquet").to_pydict()
        for camera in self.episode["cameras"]:
            frames = sorted((self.folder / "cam" / camera).glob("*.jpg"))
            if not frames:
                continue
            plain = self.folder / "preview" / f"{camera}.mp4"
            boxed = self.folder / "preview" / f"{camera}.boxes.mp4"
            plain.parent.mkdir(exist_ok=True)
            with Video(plain, self.episode["fps"]) as a, Video(boxed, self.episode["fps"]) as b:
                for row, path in enumerate(frames):
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                    a.write(img)
                    b.write(self.draw_boxes(img, camera, signals, row))
            made += [p.relative_to(self.folder).as_posix() for p in (plain, boxed)]
        return made

    def draw_boxes(self, img: Image.Image, camera: str, signals: dict, row: int) -> Image.Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        for obj, label in self.episode["labels"].items():
            key = f"{camera}.{obj}"
            if f"{key}.x0" not in signals:
                continue
            x0, y0, x1, y1 = (signals[f"{key}.{c}"][row] for c in ("x0", "y0", "x1", "y1"))
            if any(v is None or math.isnan(v) for v in (x0, y0, x1, y1)):
                continue
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BOX_COLOR, width=2)
            draw.text((x0 + 2, max(0, y0 - 12)), label, fill=BOX_COLOR)
        return img


class Video:
    """An H.264 MP4 that browsers play, written frame by frame."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.writer = path, fps, None

    def write(self, img: Image.Image) -> None:
        # H.264's usual pixel format needs even sizes.
        w, h = img.size
        if w % 2 or h % 2:
            img = img.crop((0, 0, w - w % 2, h - h % 2))
        if self.writer is None:
            self.writer = imageio_ffmpeg.write_frames(
                str(self.path), img.size, fps=self.fps, codec="libx264",
                pix_fmt_out="yuv420p", quality=None, macro_block_size=2,
                output_params=["-crf", str(QUALITY), "-preset", "medium", "-movflags", "+faststart"],
            )
            self.writer.send(None)
        self.writer.send(np.ascontiguousarray(np.asarray(img)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.writer is not None:
            self.writer.close()
