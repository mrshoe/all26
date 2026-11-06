"""Labeled photos, e.g. a public dataset of real FUEL pictures, as a run of
episodes like the simulator's, so a dataset.toml can list it (usually
under [test]) and the scorer can score a model on it.

    python -m shared.photos import ~/Downloads/fuel-real --splits valid test \\
        --labels Fuel=fuel Bumper=bumper \\
        --name "Popcorn Penguins (6238) FUEL photos, v15" \\
        --url https://universe.roboflow.com/testing-2uiab/fuel-detection/dataset/15 \\
        --license "CC BY 4.0" --credit "Popcorn Penguins, FRC 6238"

It reads a folder in ultralytics YOLO's format (data.yaml, and
<split>/images, <split>/labels, as Roboflow exports them), writes one
episode per split (each photo a frame of the camera "photo"), uploads them
under runs/<you>/<run>/, and records the run in Firestore with `photos` in
place of `sim`. It prints the run's ID.

What photos don't have, next to a simulation: masks, object ids, poses,
camera intrinsics, keypoints, full boxes, and pixel counts (min_pixels then
compares with the box's area). Labels are only what the labeler drew.

Each photo's file name is in the episode's `photos` table (frame, file,
split). Take only splits whose photos are all different pictures: a
Roboflow train split often holds several augmented copies of each.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import re
import secrets
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
from PIL import Image, ImageOps

from shared.cloud import Cloud
from shared.logger import Logger, schema

CAMERA = "photo"
IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
UPLOADS = 8
ORIENTATION = 0x0112  # EXIF
PHOTOS = {"photos": schema(("frame", pa.int32()), ("file", pa.string()), ("split", pa.string()), key=("frame",))}


def new_run_id() -> str:
    """Like wildhat launch's: 20261106-0012-k3rq."""
    alphabet = "abcdefghijkmnpqrstuvwxyz23456789"
    return datetime.datetime.now().strftime("%Y%m%d-%H%M-") + "".join(secrets.choice(alphabet) for _ in range(4))


def class_names(folder: Path) -> list[str]:
    """data.yaml's `names`, as a list or a {0: name} map (no YAML library needed)."""
    text = (folder / "data.yaml").read_text()
    found = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
    if found:
        return [n.strip().strip("'\"") for n in found.group(1).split(",") if n.strip()]
    found = re.search(r"^names:[ \t]*\n((?:(?:[ \t]+|-).*\n?)+)", text, re.M)
    if found:
        pairs = re.findall(r"^\s*(\d+)\s*:\s*(.+?)\s*$", found.group(1), re.M)
        if pairs:
            return [n.strip("'\"") for _, n in sorted(pairs, key=lambda p: int(p[0]))]
        return [line.strip().lstrip("-").strip().strip("'\"") for line in found.group(1).splitlines() if line.strip()]
    raise ValueError(f"{folder / 'data.yaml'} has no names")


def write_episode(folder: Path, out: Path, split: str, labels: dict[int, str], info: dict) -> dict:
    """One split's photos and boxes as an episode in `out`; returns episode.json."""
    images = sorted(p for p in (folder / split / "images").iterdir() if p.suffix.lower() in IMAGES)
    if not images:
        raise ValueError(f"{folder / split / 'images'} has no pictures")
    log = Logger(out, info={**info, "split": split}, schemas=PHOTOS)
    sizes: dict[tuple[int, int], int] = {}
    counts: Counter = Counter()
    for frame, path in enumerate(images):
        with Image.open(path) as img:
            turned = img.getexif().get(ORIENTATION, 1) != 1
            jpeg = img.format == "JPEG"
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        width, height = rgb.size
        if (width, height) not in sizes:
            sizes[(width, height)] = len(sizes)
        camera = sizes[(width, height)]
        if jpeg and not turned:  # as it is, not compressed again
            image = f"cam/{camera_name(camera)}/{frame:06d}.jpg"
            (out / image).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out / image)
            log.add("images", {"frame": frame, "camera": camera, "time": float(frame), "image": image})
        else:
            log.image(frame, camera, camera_name(camera), float(frame), rgb)
        log.add("photos", {"frame": frame, "file": path.name, "split": split})
        label_file = folder / split / "labels" / (path.stem + ".txt")
        for line in label_file.read_text().splitlines() if label_file.exists() else []:
            parts = line.split()
            if len(parts) != 5:
                continue  # a polygon, not a box
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            if c not in labels:
                continue
            log.add("boxes", {"frame": frame, "camera": camera, "label": labels[c], "source": "truth",
                              "x0": (cx - w / 2) * width, "y0": (cy - h / 2) * height,
                              "x1": (cx + w / 2) * width, "y1": (cy + h / 2) * height})
            counts[labels[c]] += 1
    for (width, height), camera in sizes.items():
        log.add("objects", {"object": camera, "name": camera_name(camera), "type": "Camera", "label": None,
                            "static": True, "settings": "{}"})
        log.add("cameras", {"object": camera, "width": width, "height": height})
    return log.finish({"photos": len(images), "labels": dict(counts)})


def camera_name(camera: int) -> str:
    return CAMERA if camera == 0 else f"{CAMERA}{camera + 1}"


def upload(cloud: Cloud, folder: Path, prefix: str) -> int:
    files = [p for p in folder.rglob("*") if p.is_file()]
    types = {".jpg": "image/jpeg", ".json": "application/json"}
    with concurrent.futures.ThreadPoolExecutor(UPLOADS) as pool:
        jobs = [pool.submit(cloud.upload, prefix + p.relative_to(folder).as_posix(), p,
                            types.get(p.suffix, "application/octet-stream")) for p in files]
        for job in jobs:
            job.result()
    return len(files)


def import_photos(folder, splits: list[str], labels: dict[str, str], about: dict,
                  cloud: Cloud | None = None, say=print) -> str:
    """Upload these splits of a YOLO-format folder as a new run; returns its ID.
    `labels` maps the folder's class names to ours (others are left out)."""
    folder = Path(folder).expanduser()
    cloud = cloud or Cloud()
    names = class_names(folder)
    unknown = set(labels) - set(names)
    if unknown:
        raise ValueError(f"{', '.join(sorted(unknown))} isn't a class there (they're {', '.join(names)})")
    ids = {i: labels[n] for i, n in enumerate(names) if n in labels}
    handle = cloud.handle
    run = new_run_id()
    prefix = f"runs/{handle}/{run}/"
    outputs = f"gs://{cloud.bucket}/{prefix}"
    episodes = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, split in enumerate(splits):
            out = Path(tmp) / f"episode-{index}"
            episode = write_episode(folder, out, split, ids, {"photos_from": about.get("name")})
            n = upload(cloud, out, f"{prefix}episode-{index}/")
            say(f"Episode {index}: {split}, {episode['photos']} photos, "
                + ", ".join(f"{k} {v}" for k, v in sorted(episode["labels"].items())) + f" ({n} files)")
            episodes.append(episode)
    now = datetime.datetime.now(datetime.timezone.utc)
    photos = {**about, "labels": sorted(set(ids.values())), "splits": splits, "folder": folder.name}
    cloud.create(f"runs/{run}", {"launched_by": handle, "created_at": now, "episodes": len(splits),
                                 "outputs": outputs, "photos": photos})
    for index, (split, episode) in enumerate(zip(splits, episodes)):
        cloud.create(f"runs/{run}/episodes/{index}", {
            "index": index, "status": "succeeded", "outputs": f"{outputs}episode-{index}/",
            "started_at": now, "finished_at": now,
            "photos": {"split": split, "frames": episode["photos"], "labels": sorted(episode["labels"]),
                       "cameras": [CAMERA]}})
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shared.photos", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    imp = sub.add_parser("import", help="upload a YOLO-format folder of labeled photos as a run")
    imp.add_argument("folder")
    imp.add_argument("--splits", nargs="+", required=True, help="its splits to take, one episode each")
    imp.add_argument("--labels", nargs="+", required=True, metavar="THEIRS=OURS",
                     help="its class names and ours, e.g. Fuel=fuel; others are left out")
    imp.add_argument("--name", required=True, help="what it is, for people browsing runs")
    imp.add_argument("--url", help="where it came from")
    imp.add_argument("--license")
    imp.add_argument("--credit", help="who made it")
    args = parser.parse_args(argv)
    labels = dict(pair.split("=", 1) for pair in args.labels)
    about = {k: v for k, v in {"name": args.name, "url": args.url, "license": args.license,
                               "credit": args.credit}.items() if v}
    run = import_photos(args.folder, args.splits, labels, about)
    print(f"Imported run {run}. List it in a dataset.toml, e.g. under [test]: runs = [\"{run}\"]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
