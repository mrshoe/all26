"""`PostProcessor`: after an episode, make the previews the visualizer plays,
from a folder `logger.Logger` wrote. The simulator runs it after each
episode, and the robot's upload tool before uploading.

    preview/<camera>.mp4        the camera's frames as a small H.264 video
    preview/<camera>.boxes.mp4  the same, with labels drawn: ground truth in
                                magenta, a model's in cyan. Boxes are thick;
                                a full box (around the whole object, where
                                it's partly hidden) is thin. Keypoints are
                                dots: filled if seen, hollow if hidden.

Previews are low bitrate on purpose: the visualizer streams them out of the
bucket, and every GB downloaded costs (ai/INFRA.md). The full-quality
frames are still in cam/.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from shared.episodes import Episode

# Ground truth, and a model's detections.
TRUTH_COLOR = (255, 40, 200)
DETECTION_COLOR = (40, 220, 255)
# Keypoints' dots, in pixels.
POINT_RADIUS = 3
# Higher is smaller and blurrier; 23 is ffmpeg's default.
QUALITY = 30


class PostProcessor:
    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def run(self) -> list[str]:
        """Make every camera's previews; return their paths in the folder."""
        ep = Episode(self.folder)
        if "images" not in ep.info["tables"]:
            return []
        fps = ep.info.get("fps", 10)
        boxes = defaultdict(list)  # (camera id, frame) -> rows
        if "boxes" in ep.info["tables"]:
            for row in ep.table("boxes").to_pylist():
                boxes[row["camera"], row["frame"]].append(row)
        points = defaultdict(list)  # (camera id, frame) -> rows
        if "keypoints" in ep.info["tables"]:
            for row in ep.table("keypoints").to_pylist():
                if row["in_frame"]:
                    points[row["camera"], row["frame"]].append(row)
        images = defaultdict(list)  # camera id -> (frame, path)
        for row in ep.table("images").to_pylist():
            images[row["camera"]].append((row["frame"], row["image"]))
        made = []
        for camera, frames in images.items():
            name = ep.names[camera]
            plain = self.folder / "preview" / f"{name}.mp4"
            boxed = self.folder / "preview" / f"{name}.boxes.mp4"
            plain.parent.mkdir(exist_ok=True)
            with Video(plain, fps) as a, Video(boxed, fps) as b:
                for frame, path in sorted(frames):
                    with Image.open(self.folder / path) as img:
                        img = img.convert("RGB")
                    a.write(img)
                    b.write(draw_labels(img, boxes[camera, frame], points[camera, frame]))
            made += [p.relative_to(self.folder).as_posix() for p in (plain, boxed)]
        return made


def draw_labels(img: Image.Image, boxes: list[dict], points: list[dict] = ()) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for r in boxes:
        truth = r["source"] == "truth"
        color = TRUTH_COLOR if truth else DETECTION_COLOR
        text = r["label"] if truth else f"{r['label']} {r['score']:.2f}"
        full = [r.get(f"full_{k}") for k in ("x0", "y0", "x1", "y1")]
        if None not in full and max(abs(a - b) for a, b in zip(full, (r["x0"], r["y0"], r["x1"], r["y1"]))) > 1:
            draw.rectangle([full[0], full[1], full[2] - 1, full[3] - 1], outline=color, width=1)
        draw.rectangle([r["x0"], r["y0"], r["x1"] - 1, r["y1"] - 1], outline=color, width=2)
        draw.text((r["x0"] + 2, max(0, r["y0"] - 12)), text, fill=color)
    for r in points:
        color = TRUTH_COLOR if r["source"] == "truth" else DETECTION_COLOR
        x, y, radius = r["x"], r["y"], POINT_RADIUS
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=1,
                     fill=None if r["visible"] is False else color)
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
