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

How it splits:

- By episode, never by frame: an episode's frames are nearly the same, so
  splitting frames would put training data in validation.
- An episode's split comes from a hash of its run and number, so adding runs
  never moves an episode that's already in the dataset.
- [test] runs are always "test", whole.

Only episodes that succeeded are used. A run that's still going is fetched
as far as it has got, with a warning, and the dataset grows as it finishes,
so list runs once they're done.

Downloads go to ~/.cache/wildhat/runs/<run>/episode-<n>/ (or $WILDHAT_CACHE),
without the preview videos. Once a finished run is fetched, using it needs no
network.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
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

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"There's no split {split!r}; they're {', '.join(SPLITS)}")


def _by_episode(samples: list[Sample]):
    groups: dict[int, list[Sample]] = {}
    for s in samples:
        groups.setdefault(id(s.episode), []).append(s)
    return [(g[0].episode, g) for g in groups.values()]


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

    episodes, files, total = [], [], 0
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
    parser.add_argument("action", choices=["check", "fetch"],
                        help="check: look up its runs without downloading; fetch: download them, then count")
    parser.add_argument("dataset", help="a model directory, or its dataset.toml")
    args = parser.parse_args(argv)
    try:
        ds = Dataset.load(args.dataset)
        if args.action == "check":
            return 0 if ds.check() else 1
        ds.fetch()
        for split, s in ds.summary().items():
            labels = ", ".join(f"{label} {n}" for label, n in sorted(s["labels"].items())) or "no boxes"
            print(f"{split}: {s['episodes']} episode(s), {s['samples']} sample(s); {labels}")
        return 0
    except (ValueError, KeyError, CloudError) as e:
        print(f"datasets: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
