"""A baseline AprilTag detector, OpenCV's, to score against the tags'
ground truth (`shared.score`): what a classic detector finds in a dataset's
pictures, at each distance and angle, before any model is trained.

    python -m shared.apriltags detect training/<model> --split test --out preds/opencv
    python -m shared.score preds/opencv

It finds 36h11 tags (FRC's) with OpenCV's ArUco detector, refining corners
the way the AprilTag library does, and gives each tag's corners in the
truth's order: bottom left, bottom right, top right, top left, as seen
facing the tag. (OpenCV's own order is top left first, but of its own
36h11 drawings, which are the official tags turned 180 degrees, so on a
real tag it starts at the bottom right.) Pixels are as the truth's: (0, 0)
is the picture's top-left corner, not the middle of its first pixel.

It isn't the robot's detector (robotpy_apriltag, on the Pi), whose
settings differ; it's a yardstick. OpenCV gives no confidence, so its tags
have no score.

Needs OpenCV (pip install opencv-python-headless; ultralytics brings it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODEL = "opencv_apriltag"
# OpenCV's corners (on a real tag) in the truth's order.
FROM_OPENCV = [1, 0, 3, 2]

_detector = None


def detector():
    global _detector
    if _detector is None:
        import cv2

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        _detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
    return _detector


def detect(rgb: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """The 36h11 tags in a picture (height x width x 3 uint8, RGB): each
    one's ID and its 4 corners (4 x 2 pixels), in the truth's order."""
    import cv2

    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector().detectMarkers(gray)
    if ids is None:
        return []
    return [(int(i), np.asarray(c, float).reshape(4, 2)[FROM_OPENCV]) for c, i in zip(corners, ids.ravel())]


def run(dataset, split: str, folder, model: str = MODEL, say=print) -> Path:
    """Detect tags in every picture of a dataset's split, and write them as
    predictions in `folder`."""
    from shared.datasets import Dataset
    from shared.predictions import Predictions

    if not isinstance(dataset, Dataset):
        dataset = Dataset.load(dataset)
    dataset.fetch()
    samples = dataset.samples(split)
    if not samples:
        raise ValueError(f"The dataset's {split} split has no pictures")
    preds = Predictions(folder, model=model, classes=["apriltag"])
    found = 0
    for n, sample in enumerate(samples, start=1):
        for tag, corners in detect(sample.image()):
            preds.tag(sample, tag, corners)
            found += 1
        preds.done(sample)
        if n % 200 == 0:
            say(f"{n}/{len(samples)} pictures, {found} tags")
    say(f"{len(samples)} pictures, {found} tags")
    return preds.finish(dataset, split)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shared.apriltags", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    det = sub.add_parser("detect", help="detect tags in a dataset's split, as predictions")
    det.add_argument("dataset", help="a dataset.toml, or the model directory that has one")
    det.add_argument("--split", default="validation")
    det.add_argument("--out", required=True, help="the predictions folder to write")
    det.add_argument("--model", default=MODEL, help=f"the predictions' source name (default {MODEL})")
    args = parser.parse_args(argv)
    folder = run(args.dataset, args.split, args.out, args.model)
    print(f"Wrote {folder}. Score it: python -m shared.score {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
