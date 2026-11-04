"""Datasets: which simulation runs a model trains on, and how they split.

Each model's directory has a dataset.toml:

    # ai/src/training/fuel_detector/dataset.toml
    runs = ["20261102-0019-w22f", "20261104-1830-k3rq"]
    every = 2              # use every 2nd frame (neighboring frames are nearly the same)
    validation = 0.15      # share of episodes for validation
    cameras = ["front"]    # optional: only these cameras (default: all of them)

    [test]                 # optional: runs never trained on, e.g. a held-out scene
    runs = ["20261106-0900-p7tz"]

And from ai/src:

    from shared.datasets import Dataset

    ds = Dataset.load("training/fuel_detector")   # the directory, or its dataset.toml
    ds.fetch()                                    # downloads what's missing
    for sample in ds.samples("train"):            # or "validation", "test"
        sample.image(), sample.boxes(), sample.mask()
    ds.episodes("validation")                     # shared.episodes.Episode objects

Or check one, and fetch it, from the command line (also from ai/src):

    python -m shared.datasets check training/fuel_detector
    python -m shared.datasets fetch training/fuel_detector

For ultralytics YOLO, export it in YOLO's format and train on the data.yaml:

    data_yaml = ds.yolo(classes=["fuel"], min_pixels=20)
    YOLO("yolo26n.pt").train(data=str(data_yaml), ...)

    python -m shared.datasets yolo training/fuel_detector --classes fuel

How it splits:

- By episode, never by frame: an episode's frames are nearly the same, so
  splitting frames would put training data in validation.
- An episode's split comes from a hash of its run and number, so adding runs
  never moves an episode that's already in the dataset.
- [test] runs are always "test", whole.

Only episodes that succeeded are used. A run that's still going is fetched
as far as it has got, with a warning, and the dataset grows as it finishes,
so list runs once they're done.

Only episodes in the current format are used: the first runs, from before
the episode tables (Oct 30-31, 2026), have only pictures and videos.

Downloads go to ~/.cache/wildhat/runs/<run>/episode-<n>/ (or $WILDHAT_CACHE),
without the preview videos. Once a finished run is fetched, using it needs no
network. YOLO exports go to ~/.cache/wildhat/yolo/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from shared.cloud import Cloud, CloudError
from shared.episodes import Box, Episode

RUN_ID = re.compile(r"^\d{8}-\d{4}-[a-z0-9]{4}$")
SPLITS = ("train", "validation", "test")
KEYS = {"runs", "every", "validation", "cameras", "test"}
# Training doesn't need the preview videos and contact sheets.
SKIP = ("preview/",)
FETCHED = ".fetched"
DOWNLOADS = 16
# Every episode in the current format has this; the first ones (format 1) don't.
TABLES = "objects.parquet"
# Bump when the YOLO export changes, so old exports aren't reused.
YOLO_VERSION = 1


def cache_dir() -> Path:
    return Path(os.environ.get("WILDHAT_CACHE") or Path.home() / ".cache" / "wildhat")


@dataclass(frozen=True)
class Sample:
    """One camera's picture in one frame of an episode."""

    episode: Episode
    run: str
    index: int  # the episode's number in its run
    camera: str
    frame: int

    def image(self):
        return self.episode.image(self.camera, self.frame)

    def mask(self):
        return self.episode.mask(self.camera, self.frame)

    def boxes(self) -> list[Box]:
        return self.episode.boxes(self.camera, self.frame)


def split_share(run: str, index: int) -> float:
    """Where an episode falls in [0, 1), the same on every computer, forever."""
    digest = hashlib.sha256(f"{run}/episode-{index}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


class Dataset:
    def __init__(self, path: Path, runs: list[str], test_runs: list[str], every: int, validation: float,
                 cameras: list[str] | None):
        self.path = path
        self.runs = runs
        self.test_runs = test_runs
        self.every = every
        self.validation = validation
        self.cameras = cameras

    @property
    def name(self) -> str:
        """The model directory's name, e.g. fuel_detector."""
        return self.path.resolve().parent.name

    # --- Reading dataset.toml --------------------------------------------------

    @classmethod
    def load(cls, path) -> "Dataset":
        path = Path(path)
        if path.is_dir():
            path = path / "dataset.toml"
        try:
            with path.open("rb") as f:
                config = tomllib.load(f)
        except FileNotFoundError:
            raise ValueError(f"There's no {path}") from None
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path} isn't valid TOML: {e}") from None

        def problem(message: str) -> ValueError:
            return ValueError(f"{path}: {message}")

        unknown = set(config) - KEYS
        if unknown:
            raise problem(f"unknown setting {', '.join(sorted(unknown))} (the settings are {', '.join(sorted(KEYS))})")
        test = config.get("test", {})
        if not isinstance(test, dict) or set(test) - {"runs"}:
            raise problem("[test] takes only runs = [...]")

        def run_list(value, where: str) -> list[str]:
            if not isinstance(value, list) or not all(isinstance(r, str) for r in value):
                raise problem(f"{where} must be a list of run IDs, like [\"20261102-0019-w22f\"]")
            for run in value:
                if not RUN_ID.match(run):
                    raise problem(f"{run!r} in {where} isn't a run ID. They look like 20261102-0019-w22f.")
            repeated = [r for r, n in Counter(value).items() if n > 1]
            if repeated:
                raise problem(f"{', '.join(repeated)} is listed twice in {where}")
            return value

        if "runs" not in config:
            raise problem("it needs runs = [...], the runs to train on")
        runs = run_list(config["runs"], "runs")
        if not runs:
            raise problem("runs is empty")
        test_runs = run_list(test.get("runs", []), "[test] runs")
        both = sorted(set(runs) & set(test_runs))
        if both:
            raise problem(f"{', '.join(both)} is in both runs and [test]; a test run must never be trained on")

        every = config.get("every", 1)
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise problem("every must be a whole number, 1 or more")
        validation = config.get("validation", 0.15)
        if not isinstance(validation, (int, float)) or isinstance(validation, bool) or not 0 <= validation < 1:
            raise problem("validation must be a share from 0 up to (not including) 1, like 0.15")
        cameras = config.get("cameras")
        if cameras is not None and (not isinstance(cameras, list) or not cameras
                                    or not all(isinstance(c, str) for c in cameras)):
            raise problem("cameras must be a list of camera names, like [\"front\"]")
        return cls(path, runs, test_runs, every, float(validation), cameras)

    def split_of(self, run: str, index: int) -> str:
        if run in self.test_runs:
            return "test"
        return "validation" if split_share(run, index) < self.validation else "train"

    # --- Downloading -------------------------------------------------------------

    def fetch(self, cloud: Cloud | None = None, say=None) -> "Dataset":
        """Download every run's succeeded episodes that aren't cached yet."""
        cloud, say = cloud or Cloud(), say or print
        for run in self.runs + self.test_runs:
            fetch_run(cloud, run, say)
        return self

    def check(self, cloud: Cloud | None = None, say=None) -> bool:
        """Look up every run without downloading anything: say what's wrong,
        and how the episodes would split. True if nothing's wrong (warnings
        don't count)."""
        cloud, say = cloud or Cloud(), say or print
        ok = True
        counts = Counter()
        for run in self.runs + self.test_runs:
            record = cloud.document(f"runs/{run}")
            if record is None:
                say(f"✗ {run}: there's no such run")
                ok = False
                continue
            statuses = {int(e.get("index", i)): e.get("status") for i, e in cloud.documents(f"runs/{run}/episodes").items()}
            good = sorted(i for i, s in statuses.items() if s == "succeeded")
            others = Counter(s for s in statuses.values() if s != "succeeded")
            note = "" if not others else " (" + ", ".join(f"{n} {s}" for s, n in sorted(others.items())) + ")"
            if not good:
                say(f"✗ {run}: no episode succeeded{note}")
                ok = False
                continue
            prefix = record["outputs"].split("/", 3)[3]
            if not cloud.objects(f"{prefix}episode-{good[0]}/{TABLES}"):
                say(f"✗ {run}: it's in the old format, from before the episode tables, so it can't be trained on")
                ok = False
                continue
            mark = "!" if others else "✓"
            say(f"{mark} {run}: {len(good)} episode(s) succeeded{note}, by {record.get('launched_by', '?')}")
            if others.get("pending") or others.get("running"):
                say("  It's still going. The dataset will grow as it finishes; best to list runs once they're done.")
            for i in good:
                counts[self.split_of(run, i)] += 1
        say("Episodes: " + ", ".join(f"{split} {counts[split]}" for split in SPLITS))
        if not counts["train"]:
            say("✗ No training episodes.")
            ok = False
        if self.validation and not counts["validation"]:
            say("! No validation episodes. Add runs, or raise validation.")
        return ok

    # --- Using it ------------------------------------------------------------------

    @cached_property
    def _episodes(self) -> list[tuple[str, int, Episode]]:
        out = []
        for run in self.runs + self.test_runs:
            manifest = cache_dir() / "runs" / run / "run.json"
            if not manifest.exists():
                raise CloudError(f"Run {run} isn't downloaded yet. Call fetch() first, or run "
                                 f"python -m shared.datasets fetch {self.path}")
            for index in json.loads(manifest.read_text())["episodes"]:
                out.append((run, index, Episode(manifest.parent / f"episode-{index}")))
        return out

    def episodes(self, split: str) -> list[Episode]:
        self._check_split(split)
        return [ep for run, index, ep in self._episodes if self.split_of(run, index) == split]

    def episode_ids(self, split: str) -> list[str]:
        """The split's episodes as "<run>/<number>", e.g. for a record of
        exactly what a model trained on."""
        self._check_split(split)
        return [f"{run}/{index}" for run, index, _ in self._episodes if self.split_of(run, index) == split]

    def samples(self, split: str) -> list[Sample]:
        """Every picture in the split: each chosen camera, every `every`th frame."""
        self._check_split(split)
        out = []
        for run, index, ep in self._episodes:
            if self.split_of(run, index) != split:
                continue
            cameras = self.cameras or ep.cameras
            missing = set(cameras) - set(ep.cameras)
            if missing:
                raise KeyError(f"{run} episode {index} has no camera {', '.join(sorted(missing))} "
                               f"(it has {', '.join(ep.cameras) or 'none'})")
            ids = {ep.objects[c]["object"]: c for c in cameras}
            images = ep.table("images")
            for frame, camera in zip(images["frame"].to_pylist(), images["camera"].to_pylist()):
                if camera in ids and frame % self.every == 0:
                    out.append(Sample(ep, run, index, ids[camera], frame))
        return out

    def summary(self) -> dict[str, dict]:
        """For each split: how many episodes and samples, and how many of each
        label the samples' ground-truth boxes hold."""
        out = {}
        for split in SPLITS:
            samples = self.samples(split)
            labels = Counter()
            for ep, group in _by_episode(samples):
                if "boxes" not in ep.info["tables"]:
                    continue
                chosen = {(ep.objects[s.camera]["object"], s.frame) for s in group}
                t = ep.table("boxes").to_pydict()
                for frame, camera, label, source in zip(t["frame"], t["camera"], t["label"], t["source"]):
                    if source == "truth" and (camera, frame) in chosen:
                        labels[label] += 1
            out[split] = {"episodes": len(self.episodes(split)), "samples": len(samples), "labels": dict(labels)}
        return out

    # --- Exporting ---------------------------------------------------------------

    def yolo(self, classes: list[str], min_pixels: int = 0, say=None) -> Path:
        """Export the dataset in ultralytics YOLO's detection format, and
        return its data.yaml, for YOLO(...).train(data=str(path)).

        `classes` are the labels to detect, in order: the first is class 0.
        Boxes with other labels are left out, and so are boxes covering fewer
        than `min_pixels` pixels (objects almost entirely hidden). Pictures
        with no boxes left stay in, with empty label files: they teach the
        model what "nothing here" looks like.

        It goes to ~/.cache/wildhat/yolo/<model>-<hash>/, never into ai/src,
        with the pictures hard-linked from the cache (copied if they can't
        be). The hash covers the episodes, every, cameras and these options,
        so exporting the same dataset again reuses what's there."""
        say = say or print
        if (not isinstance(classes, (list, tuple)) or not classes
                or not all(isinstance(c, str) and c for c in classes)):
            raise ValueError("classes must be a list of labels, like [\"ball\"]")
        repeated = [c for c, n in Counter(classes).items() if n > 1]
        if repeated:
            raise ValueError(f"{', '.join(repeated)} is in classes twice")
        if not isinstance(min_pixels, int) or isinstance(min_pixels, bool) or min_pixels < 0:
            raise ValueError("min_pixels must be a whole number, 0 or more")

        splits = {split: self.episode_ids(split) for split in SPLITS}
        if not splits["train"]:
            raise ValueError("There are no training episodes to export.")
        if not splits["validation"]:
            raise ValueError("There are no validation episodes, and YOLO needs some. "
                             "Add runs, or raise validation in dataset.toml.")
        key = json.dumps({"version": YOLO_VERSION, "splits": splits, "every": self.every, "cameras": self.cameras,
                          "classes": list(classes), "min_pixels": min_pixels}, sort_keys=True)
        folder = cache_dir() / "yolo" / f"{self.name}-{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        data_yaml = folder / "data.yaml"
        if data_yaml.exists():
            say(f"YOLO dataset: {folder} (already exported)")
            return data_yaml

        # Build it next to where it goes, then move it there in one step, so
        # a crash halfway never leaves an export that looks finished.
        building = folder.with_name(f"{folder.name}.building-{os.getpid()}")
        shutil.rmtree(building, ignore_errors=True)
        ids = {label: n for n, label in enumerate(classes)}
        counts = {}
        for split in SPLITS:
            images, labels = building / "images" / split, building / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            pictures = boxes = 0
            for ep, group in _by_episode(self.samples(split)):
                sizes = {r["object"]: (r["width"], r["height"]) for r in ep.table("cameras").to_pylist()}
                files = {(r["camera"], r["frame"]): r["image"] for r in ep.table("images").to_pylist()}
                lines: dict[tuple[int, int], list[str]] = {}
                if "boxes" in ep.info["tables"]:
                    for r in ep.table("boxes").to_pylist():
                        if r["source"] != "truth" or r["label"] not in ids or (r["pixels"] or 0) < min_pixels:
                            continue
                        line = _yolo_line(ids[r["label"]], r, *sizes[r["camera"]])
                        if line:
                            lines.setdefault((r["camera"], r["frame"]), []).append(line)
                for s in group:
                    camera = ep.objects[s.camera]["object"]
                    stem = f"{s.run}_{s.index}_{s.camera}_{s.frame:06d}"
                    source = ep.folder / files[(camera, s.frame)]
                    _link(source, images / f"{stem}{source.suffix}")
                    found = lines.get((camera, s.frame), [])
                    (labels / f"{stem}.txt").write_text("".join(f"{line}\n" for line in found))
                    pictures += 1
                    boxes += len(found)
            counts[split] = (pictures, boxes)

        names = "".join(f"  {n}: {json.dumps(label)}\n" for n, label in enumerate(classes))
        test = "test: images/test\n" if counts["test"][0] else ""
        (building / "data.yaml").write_text(
            f"# {self.path}, exported by shared.datasets for ultralytics YOLO\n"
            f"path: {json.dumps(str(folder))}\ntrain: images/train\nval: images/validation\n{test}"
            f"names:\n{names}")
        try:
            building.rename(folder)
        except OSError:  # someone else just finished the same export
            shutil.rmtree(building, ignore_errors=True)
            if not data_yaml.exists():
                raise
        say(f"YOLO dataset: {folder}")
        say("  " + ", ".join(f"{split} {n} picture(s), {b} box(es)" for split, (n, b) in counts.items()))
        if not counts["train"][1]:
            say(f"  ! No training boxes labeled {', '.join(classes)}. Is that the right label?")
        return data_yaml

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"There's no split {split!r}; they're {', '.join(SPLITS)}")


def _by_episode(samples: list[Sample]):
    groups: dict[int, list[Sample]] = {}
    for s in samples:
        groups.setdefault(id(s.episode), []).append(s)
    return [(g[0].episode, g) for g in groups.values()]


def _yolo_line(class_id: int, box: dict, width: int, height: int) -> str | None:
    """`class cx cy w h`, as shares of the picture's size, clipped to it."""
    x0, x1 = max(box["x0"], 0), min(box["x1"], width)
    y0, y1 = max(box["y0"], 0), min(box["y1"], height)
    if x1 <= x0 or y1 <= y0:
        return None
    return (f"{class_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")


def _link(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:  # another disk, or a filesystem without hard links
        shutil.copyfile(source, target)


def fetch_run(cloud: Cloud, run: str, say) -> None:
    """Download a run's succeeded episodes into the cache, skipping what's
    already there, and write its run.json last."""
    folder = cache_dir() / "runs" / run
    manifest = folder / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("finished"):
        return  # nothing more can come
    record = cloud.document(f"runs/{run}")
    if record is None:
        raise CloudError(f"There's no run {run}.")
    statuses = {int(e.get("index", i)): e.get("status") for i, e in cloud.documents(f"runs/{run}/episodes").items()}
    unfinished = sorted(i for i, s in statuses.items() if s in ("pending", "running"))
    succeeded = sorted(i for i, s in statuses.items() if s == "succeeded")
    prefix = record["outputs"].split("/", 3)[3]  # gs://<bucket>/runs/<handle>/<run>/ -> runs/<handle>/<run>/

    episodes, files, total, old = [], [], 0, 0
    for index in succeeded:
        where = folder / f"episode-{index}"
        if (where / FETCHED).exists():
            episodes.append(index)
            continue
        objects = [(name, n) for name, n in cloud.objects(f"{prefix}episode-{index}/")
                   if not name.removeprefix(f"{prefix}episode-{index}/").startswith(SKIP)]
        if not any(name.endswith("/episode.json") for name, _ in objects):
            say(f"{run} episode {index} has no episode.json (not a simulation?), so it's left out")
            continue
        if not any(name == f"{prefix}episode-{index}/{TABLES}" for name, _ in objects):
            old += 1
            continue
        episodes.append(index)
        for name, n in objects:
            files.append((name, where / name.removeprefix(f"{prefix}episode-{index}/")))
            total += n

    if files:
        say(f"{run}: downloading {len(files)} file(s), {total / 1e6:.1f} MB")
        with concurrent.futures.ThreadPoolExecutor(DOWNLOADS) as pool:
            for future in [pool.submit(cloud.download, name, path) for name, path in files]:
                future.result()
    for index in episodes:
        (folder / f"episode-{index}" / FETCHED).touch()

    if old:
        say(f"{run}: {old} episode(s) are in the old format, from before the episode tables, so they're left out")
    failed = len(statuses) - len(succeeded) - len(unfinished)
    if failed:
        say(f"{run}: {failed} episode(s) failed or stopped, so they're left out")
    if unfinished:
        say(f"{run}: still going ({len(unfinished)} episode(s) not done); fetch again later for the rest")
    folder.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"episodes": episodes, "finished": not unfinished,
                                    "launched_by": record.get("launched_by")}, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shared.datasets", description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["check", "fetch", "yolo"],
                        help="check: look up its runs without downloading; fetch: download them, then count; "
                             "yolo: fetch, then export it for ultralytics YOLO")
    parser.add_argument("dataset", help="a model directory, or its dataset.toml")
    parser.add_argument("--classes", nargs="+", metavar="LABEL",
                        help="yolo: the labels to detect, in order (the first is class 0)")
    parser.add_argument("--min-pixels", type=int, default=0, metavar="N",
                        help="yolo: leave out boxes covering fewer than N pixels")
    args = parser.parse_args(argv)
    if args.action == "yolo" and not args.classes:
        parser.error("yolo needs --classes, e.g. --classes ball")
    try:
        ds = Dataset.load(args.dataset)
        if args.action == "check":
            return 0 if ds.check() else 1
        ds.fetch()
        if args.action == "yolo":
            print(ds.yolo(args.classes, args.min_pixels))
            return 0
        for split, s in ds.summary().items():
            labels = ", ".join(f"{label} {n}" for label, n in sorted(s["labels"].items())) or "no boxes"
            print(f"{split}: {s['episodes']} episode(s), {s['samples']} sample(s); {labels}")
        return 0
    except (ValueError, KeyError, CloudError) as e:
        print(f"datasets: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
