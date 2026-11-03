"""Training runs: a record of every model trained, wherever it ran, with its
checkpoints in the cloud for anyone to build on.

    from shared.training import TrainingRun

    with TrainingRun.start("fuel_detector") as run:   # ai/src/training/fuel_detector/
        run.config          # train.toml as a dict; your own settings are yours to read
        run.dataset         # its Dataset (shared/datasets.py), already fetched
        run.folder          # a local folder for your checkpoint files
        weights = run.start_weights()                # train.toml's [start], downloaded
        for step in ...:
            run.log(step, loss=loss, lr=lr)
            run.save(run.folder / "model.pt", step, metrics={"val_ap50": ap}, best=True)

Leaving the `with` block finishes the run: "finished", "failed" (an
exception, recorded with the run) or "stopped" (Ctrl-C, or the cloud's time
limit). Without `with`, call run.finish(); a script that just ends is
finished too.

A model's train.toml (optional) holds its settings. The tool reads two:

    dataset = "dataset.toml"         # default; may be another model's, e.g. "../reef_keypoints/dataset.toml"

    [start]                          # checkpoints to start from, by any name you like
    backbone = "fuel_detector:v2"    # a tagged checkpoint
    reef = "20261120-1904-q2mz"      # a training run's best checkpoint (or its last, if none was best)
    old = "20261120-1904-q2mz/4000"  # its checkpoint at step 4000

The rest is yours (freeze or not, learning rates, epochs...): run.config has
it all, and the record keeps the whole file.

Every run is recorded in Firestore as training/<id> (ids look like a run's,
20261110-2210-ab12), with its source, both files, the exact episodes of
each split, the exact checkpoints [start] resolved to, its latest metrics
and its checkpoints. Files go to the bucket under
training/<handle>/<id>/: checkpoints/<step>/<file> and metrics/*.jsonl.

On a laptop, the source is packed and uploaded the way `wildhat launch`
does it (uncommitted edits too), with `wildhat pack-source`; in a cloud job
it's the run's own.
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from shared.cloud import Cloud, CloudError, in_cloud_job, now
from shared.datasets import SPLITS, Dataset, cache_dir

SRC = Path(__file__).resolve().parent.parent  # ai/src
TRAINING = SRC / "training"
MODEL = re.compile(r"^[a-z_][a-z0-9_]*$")
RUN_ID = r"\d{8}-\d{4}-[a-z0-9]{4}"
REF_RUN = re.compile(rf"^({RUN_ID})$")
REF_STEP = re.compile(rf"^({RUN_ID})/(\d+)$")
REF_TAG = re.compile(r"^([a-z_][a-z0-9_]*):([A-Za-z0-9._-]+)$")
TOOL_KEYS = {"dataset", "start"}
# How often log() sends metrics, at most.
FLUSH_SECONDS = 30


def new_id() -> str:
    alphabet = "abcdefghijkmnpqrstuvwxyz23456789"  # No l/1 or o/0 to misread.
    return datetime.datetime.now().strftime("%Y%m%d-%H%M-") + "".join(secrets.choice(alphabet) for _ in range(4))


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_of(cloud: Cloud, say) -> dict:
    """The code this run is: in a cloud job, the run's own source; on a
    laptop, ai/src packed and uploaded by `wildhat pack-source`."""
    if in_cloud_job():
        run_id = os.environ["WILDHAT_RUN_DOC"].rsplit("/", 1)[1]
        record = cloud.document(f"runs/{run_id}") or {}
        return {"source": os.environ["WILDHAT_SOURCE"], "source_sha256": record.get("source_sha256"),
                "git": record.get("git")}
    result = subprocess.run(["wildhat", "pack-source"], cwd=SRC, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if "invalid choice: 'pack-source'" in message:
            raise CloudError("Your wildhat is too old for this; upgrade it (ai/wildhat/README.md, \"Upgrade\").")
        raise CloudError(f"Couldn't upload the code: {message}")
    reply = json.loads(result.stdout)
    if reply["untracked"]:
        # The whole team can read it, as with launch.
        say(f"Included {len(reply['untracked'])} file(s) git doesn't track yet: " + ", ".join(reply["untracked"]))
    return {k: reply[k] for k in ("source", "source_sha256", "git")}


def where() -> dict:
    if in_cloud_job():
        return {"kind": "cloud", "run": os.environ["WILDHAT_RUN_DOC"].rsplit("/", 1)[1],
                "episode": int(os.environ.get("WILDHAT_EPISODE", "0"))}
    return {"kind": "laptop", "host": socket.gethostname()}


def resolve(cloud: Cloud, ref: str) -> dict:
    """One exact checkpoint for a reference, as its record says."""
    tag = REF_TAG.match(ref)
    if tag:
        model, name = tag.groups()
        target = ((cloud.document(f"models/{model}") or {}).get("tags") or {}).get(name)
        if not target:
            raise ValueError(f"No checkpoint is tagged {ref}")
        return {**resolve(cloud, target), "ref": ref}
    whole, one = REF_RUN.match(ref), REF_STEP.match(ref)
    if not (whole or one):
        raise ValueError(f"{ref!r} isn't a checkpoint. Use a training run (20261110-2210-ab12), one of its "
                         "steps (20261110-2210-ab12/4000) or a tag (fuel_detector:v2).")
    run_id = (whole or one).group(1)
    record = cloud.document(f"training/{run_id}")
    if record is None:
        raise ValueError(f"There's no training run {run_id}")
    if one:
        step = int(one.group(2))
    else:
        step = (record.get("best") or {}).get("step")
        step = record.get("last_checkpoint") if step is None else step
        if step is None:
            raise ValueError(f"Training run {run_id} has no checkpoints yet")
    checkpoint = cloud.document(f"training/{run_id}/checkpoints/{step}")
    if checkpoint is None:
        raise ValueError(f"Training run {run_id} has no checkpoint at step {step}")
    return {"ref": ref, "checkpoint": f"{run_id}/{step}", "model": record.get("model"),
            "file": checkpoint["file"], "md5": checkpoint["md5"]}


def read_config(model: str) -> tuple[Path, dict, str | None]:
    """The model's directory, its train.toml as a dict, and the file's text
    (None if it has none)."""
    if not MODEL.match(model):
        raise ValueError(f"{model!r} isn't a model name. It's a directory in ai/src/training, "
                         "named like fuel_detector.")
    folder = TRAINING / model
    if not folder.is_dir():
        found = sorted(p.name for p in TRAINING.iterdir() if p.is_dir() and MODEL.match(p.name)) \
            if TRAINING.is_dir() else []
        raise ValueError(f"There's no ai/src/training/{model}/. The models are: {', '.join(found) or 'none yet'}")
    path = folder / "train.toml"
    if not path.exists():
        return folder, {}, None
    text = path.read_text()
    try:
        config = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path} isn't valid TOML: {e}") from None
    if not isinstance(config.get("dataset", ""), str):
        raise ValueError(f"{path}: dataset must be a path, like \"dataset.toml\"")
    start = config.get("start", {})
    if not isinstance(start, dict) or not all(isinstance(v, str) for v in start.values()):
        raise ValueError(f"{path}: [start] holds names = \"checkpoints\", like backbone = \"fuel_detector:v2\"")
    return folder, config, text


class TrainingRun:
    def __init__(self, cloud: Cloud, run_id: str, model: str, config: dict, dataset: Dataset,
                 parents: dict[str, dict], say):
        self.cloud = cloud
        self.id = run_id
        self.model = model
        self.config = config
        self.dataset = dataset
        self.parents = parents
        self.say = say
        self.prefix = f"training/{cloud.handle}/{run_id}/"
        self.folder = cache_dir() / "training" / run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        self._latest: dict[str, float] = {}
        self._step: int | None = None
        self._sent = 0  # metrics files so far
        self._flushed: float | None = None
        self._steps_saved: set[int] = set()
        self._warned = False
        self._done = False
        self._lock = threading.Lock()

    @property
    def record(self) -> str:
        return f"training/{self.id}"

    # --- Starting ---------------------------------------------------------------

    @classmethod
    def start(cls, model: str, cloud: Cloud | None = None, say=None) -> "TrainingRun":
        """Record a new training run of ai/src/training/<model>/, after
        fetching its dataset and finding its [start] checkpoints."""
        cloud, say = cloud or Cloud(), say or print
        folder, config, text = read_config(model)
        dataset_path = (folder / config.get("dataset", "dataset.toml")).resolve()
        if not dataset_path.is_relative_to(SRC.resolve()):
            raise ValueError(f"{folder / 'train.toml'}: the dataset must be in ai/src, which the cloud gets")
        dataset = Dataset.load(dataset_path).fetch(cloud, say)
        parents = {name: resolve(cloud, ref) for name, ref in config.get("start", {}).items()}
        source = source_of(cloud, say)

        run_id = new_id()
        started = now()
        cloud.create(f"training/{run_id}", {
            "model": model,
            "trained_by": cloud.handle,
            "created_at": started,
            "status": "running",
            "heartbeat_at": started,
            "finished_at": None,
            "error": None,
            "where": where(),
            "segments": [{**source, "started_at": started}],
            "config": {"path": f"training/{model}/train.toml", "toml": text},
            "dataset": {
                "path": dataset_path.relative_to(SRC.resolve()).as_posix(),
                "toml": dataset_path.read_text(),
                "episodes": {split: dataset.episode_ids(split) for split in SPLITS},
                "samples": {split: len(dataset.samples(split)) for split in SPLITS},
            },
            "parents": parents,
            "step": None,
            "latest": {},
            "best": None,
            "last_checkpoint": None,
            "checkpoints": 0,
        })
        run = cls(cloud, run_id, model, config, dataset, parents, say)
        run._watch_exit()
        counts = ", ".join(f"{s} {len(dataset.samples(s))}" for s in SPLITS)
        say(f"Training run {run_id}: {model}, by {cloud.handle}. Samples: {counts}.")
        return run

    def start_weights(self) -> dict[str, Path]:
        """Each [start] checkpoint's file, downloaded (once) and checked."""
        out = {}
        for name, parent in self.parents.items():
            path = cache_dir() / "checkpoints" / parent["checkpoint"] / Path(parent["file"]).name
            if not path.exists() or md5_of(path) != parent["md5"]:
                self.say(f"Downloading {parent['ref']} ({parent['checkpoint']}) for {name}")
                self.cloud.download(parent["file"], path)
                if md5_of(path) != parent["md5"]:
                    path.unlink()
                    raise CloudError(f"{parent['checkpoint']} didn't download intact; try again")
            out[name] = path
        return out

    # --- While it trains ------------------------------------------------------------

    def log(self, step: int, **metrics) -> None:
        """Record metrics at a step. They're sent every 30 seconds or so (and
        the first time at once), and printed then too."""
        row = {"step": int(step), "time": time.time()}
        for name, value in metrics.items():
            try:
                row[name] = float(value)  # tensors and numpy numbers too
            except (TypeError, ValueError):
                raise TypeError(f"{name} must be a number, not {type(value).__name__}") from None
        with self._lock:
            self._rows.append(row)
            self._latest.update({k: v for k, v in row.items() if k not in ("step", "time")})
            self._step = int(step)
        if self._flushed is None or time.monotonic() - self._flushed >= FLUSH_SECONDS:
            self.flush()

    def flush(self) -> None:
        """Send the metrics logged so far. A failure (offline, say) is
        printed once, and they're sent with the next ones."""
        with self._lock:
            rows, latest, step = list(self._rows), dict(self._latest), self._step
        self._flushed = time.monotonic()
        if not rows:
            return
        try:
            body = "".join(json.dumps(r) + "\n" for r in rows).encode()
            self.cloud.upload(f"{self.prefix}metrics/{self._sent:06d}.jsonl", body, "application/x-ndjson")
            self.cloud.update(self.record, {"latest": latest, "step": step, "heartbeat_at": now()})
        except (CloudError, OSError) as e:
            if not self._warned:
                self.say(f"Couldn't send metrics ({e}); they'll go with the next ones.")
                self._warned = True
            return
        self._sent += 1
        with self._lock:
            del self._rows[:len(rows)]
        self.say(f"{self.model} {self.id} step {step}: " + "  ".join(f"{k} {v:.4g}" for k, v in latest.items()))

    def save(self, path, step: int, *, resumable: bool = False, metrics: dict | None = None,
             best: bool = False) -> str:
        """Upload a checkpoint file (whatever you saved in it) and record it.
        `best=True` makes it the run's best, which a [start] reference to the
        run means. Returns its id, "<run>/<step>"."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"There's no checkpoint file {path}")
        step = int(step)
        metrics = {k: float(v) for k, v in (metrics or {}).items()}
        self.flush()
        name = f"{self.prefix}checkpoints/{step}/{path.name}"
        self.cloud.upload(name, path)
        self.cloud.update(f"{self.record}/checkpoints/{step}", {
            "step": step, "file": name, "size": path.stat().st_size, "md5": md5_of(path),
            "resumable": resumable, "metrics": metrics, "best": best, "saved_at": now(), "tags": [],
        })
        self._steps_saved.add(step)
        fields = {"last_checkpoint": step, "checkpoints": len(self._steps_saved), "heartbeat_at": now()}
        if best:
            fields["best"] = {"step": step, "metrics": metrics}
        self.cloud.update(self.record, fields)
        return f"{self.id}/{step}"

    # --- Ending ------------------------------------------------------------------------

    def finish(self, status: str = "finished", error: str | None = None) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.flush()
            self.cloud.update(self.record, {"status": status, "finished_at": now(), "heartbeat_at": now(),
                                            "error": error})
        except (CloudError, OSError) as e:
            self.say(f"Couldn't record that training run {self.id} {status}: {e}")
            return
        self.say(f"Training run {self.id} {status}.")

    def __enter__(self) -> "TrainingRun":
        return self

    def __exit__(self, kind, error, traceback) -> bool:
        if kind is None or (issubclass(kind, SystemExit) and error.code in (0, None)):
            self.finish()
        elif issubclass(kind, (KeyboardInterrupt, SystemExit)):
            self.finish("stopped")
        else:
            self.finish("failed", f"{kind.__name__}: {error}")
        return False

    def _watch_exit(self) -> None:
        """If the script ends without finishing the run: an uncaught exception
        means failed, SIGTERM (the cloud's time limit, or `wildhat cancel`)
        means stopped, and otherwise it finished."""
        ending = {"status": "finished", "error": None}
        previous_hook = sys.excepthook

        def excepthook(kind, error, traceback):
            if issubclass(kind, KeyboardInterrupt):
                ending.update(status="stopped")
            else:
                ending.update(status="failed", error=f"{kind.__name__}: {error}")
            previous_hook(kind, error, traceback)

        sys.excepthook = excepthook
        if threading.current_thread() is threading.main_thread() \
                and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            def on_term(signum, frame):
                ending.update(status="stopped")
                sys.exit(128 + signum)

            signal.signal(signal.SIGTERM, on_term)
        atexit.register(lambda: self.finish(ending["status"], ending["error"]))
