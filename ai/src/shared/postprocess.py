"""`PostProcessor`: after an episode, make the previews the visualizer plays,
from a folder `logger.Logger` wrote. The simulator runs it after each
episode, and the robot's upload tool before uploading.

    preview/<camera>.mp4        the camera's frames as a small H.264 video

The visualizer draws boxes and keypoints over it in the browser, from the
episode's tables. (Episodes from before the tables also have a
<camera>.boxes.mp4, with boxes drawn in.)

Previews are low bitrate on purpose: the visualizer streams them out of the
bucket, and every GB downloaded costs (ai/INFRA.md). The full-quality
frames are still in cam/.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from shared.episodes import Episode

# Higher is smaller and blurrier; 23 is ffmpeg's default.
QUALITY = 30


class PostProcessor:
    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def run(self) -> list[str]:
        """Make every camera's preview; return their paths in the folder."""
        ep = Episode(self.folder)
        if "images" not in ep.info["tables"]:
            return []
        fps = ep.info.get("fps", 10)
        images = defaultdict(list)  # camera id -> (frame, path)
        for row in ep.table("images").to_pylist():
            images[row["camera"]].append((row["frame"], row["image"]))
        made = []
        for camera, frames in images.items():
            preview = self.folder / "preview" / f"{ep.names[camera]}.mp4"
            preview.parent.mkdir(exist_ok=True)
            with Video(preview, fps) as video:
                for frame, path in sorted(frames):
                    with Image.open(self.folder / path) as img:
                        video.write(img.convert("RGB"))
            made.append(preview.relative_to(self.folder).as_posix())
        return made


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
