"""Training runs: a record of every model trained, wherever it ran, with its
checkpoints in the cloud for anyone to build on.

    from shared.training import TrainingRun

    with TrainingRun.start("fuel_detector", resume=args.resume) as run:   # ai/src/training/fuel_detector/
        run.config          # train.toml as a dict; your own settings are yours to read
        run.dataset         # its Dataset (shared/datasets.py), already fetched
        run.folder          # a local folder for your checkpoint files
        weights = run.start_weights()                # train.toml's [start], downloaded
        if run.resume_from:                          # resuming: your last resumable checkpoint
            ... load it, and carry on from run.resume_step
        for step in ...:
            run.log(step, loss=loss, lr=lr)
            run.save(run.folder / "model.pt", step, resumable=True, metrics={"val_ap50": ap}, best=True)

Leaving the `with` block finishes the run: "finished", "failed" (an
exception, recorded with the run) or "stopped" (Ctrl-C, or the cloud's time
limit). Without `with`, call run.finish(); a script that just ends is
finished too.

A model's train.toml (optional) holds its settings. The tool reads two:

    dataset = "dataset.toml"         # default; may be another model's, e.g. "../reef_keypoints/dataset.toml"

    [start]                          # checkpoints to start from, by any name you like
    backbone = "fuel_detector:v2"    # a tagged checkpoint (wildhat training tag)
    reef = "20261120-1904-q2mz"      # a training run's best checkpoint (or its last, if none was best)
    old = "20261120-1904-q2mz/4000"  # its checkpoint at step 4000

The rest is yours (freeze or not, learning rates, epochs...): run.config has
it all, and the record keeps the whole file.

Starting from a checkpoint ([start]) begins a new training run with those
weights. Resuming (resume="<id>", usually from a --resume option) carries on
the same run where it stopped: the 8-hour limit, a crash, a laptop that
slept. Put whatever you need to carry on in a resumable checkpoint (the
optimizer's state and the step, at least).

Every run is recorded in Firestore as training/<id> (ids look like a run's,
20261110-2210-ab12), with its source, both files, the exact episodes of
each split, the exact checkpoints [start] resolved to, its latest metrics
and its checkpoints. Files go to the bucket under training/<handle>/<id>/:
checkpoints/<step>/<file> and metrics/*.jsonl.

Uploads happen in the background, so training never waits on them, in the
order they were made. Each one waits in a queue folder,
~/.cache/wildhat/training/<id>/queue/, until it's sent: offline, they wait
and are retried; if the script ends before they're all sent (on a laptop,
after a minute of failing), `wildhat training sync` sends the rest later.

A run keeps its newest 3 resumable checkpoints, its best one, any that are
tagged, and every one that isn't resumable. Older resumable ones are
deleted from the bucket (their records stay, marked pruned).

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
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from shared.cloud import Cloud, CloudError, in_cloud_job, now, patch_body
from shared.datasets import SPLITS, Dataset, cache_dir

SRC = Path(__file__).resolve().parent.parent  # ai/src
TRAINING = SRC / "training"
MODEL = re.compile(r"^[a-z_][a-z0-9_]*$")
RUN_ID = r"\d{8}-\d{4}-[a-z0-9]{4}"
REF_RUN = re.compile(rf"^({RUN_ID})$")
REF_STEP = re.compile(rf"^({RUN_ID})/(\d+)$")
REF_TAG = re.compile(r"^([a-z_][a-z0-9_]*):([A-Za-z0-9._-]+)$")
# How often log() sends metrics, at most.
FLUSH_SECONDS = 30
# How often a running run says it's alive, and how long without that before
# it's taken to have stopped. `wildhat training` uses the same.
HEARTBEAT_SECONDS = 60
STALE_SECONDS = 180
# How long a failing upload waits before trying again.
RETRY_SECONDS = 15
# On a laptop, how long finishing keeps trying uploads that fail before
# leaving them for `wildhat training sync`. A cloud job keeps trying: its
# files go when it ends.
GIVE_UP_SECONDS = 60
# Resumable checkpoints kept per run (plus its best and tagged ones).
KEEP_RESUMABLE = 3
# The queue folder's lock is touched this often while a run is sending, and
# `wildhat training sync` leaves a queue alone if its lock is newer than
# LOCK_SECONDS.
TOUCH_SECONDS = 20
LOCK_SECONDS = 120


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


def parse_time(stamp) -> datetime.datetime | None:
    """A record's time: a datetime, or the RFC 3339 text Firestore gives."""
    if stamp is None or isinstance(stamp, datetime.datetime):
        return stamp
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)", stamp)
    if not m:
        return None
    when = datetime.datetime.fromisoformat(m.group(1) + ("+00:00" if m.group(3) == "Z" else m.group(3)))
    return when.replace(microsecond=int((m.group(2) or "0")[:6].ljust(6, "0")))


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
    if checkpoint.get("pruned"):
        raise ValueError(f"{run_id}/{step} was deleted to save space: a run keeps only its newest "
                         f"{KEEP_RESUMABLE} resumable checkpoints, its best, and tagged ones")
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


def download_checkpoint(cloud: Cloud, checkpoint: str, file: str, md5: str, say) -> Path:
    """A checkpoint's file in the cache, downloaded once and checked."""
    path = cache_dir() / "checkpoints" / checkpoint / Path(file).name
    if not path.exists() or md5_of(path) != md5:
        say(f"Downloading {checkpoint}")
        cloud.download(file, path)
        if md5_of(path) != md5:
            path.unlink()
            raise CloudError(f"{checkpoint} didn't download intact; try again")
    return path


# --- The upload queue ------------------------------------------------------------
#
# A queue folder holds jobs, numbered in order: <n>.json, each
#
#   {"uploads": [{"object": "training/...", "file": "<a file in the queue folder>"}
#                or {"object": "training/...", "text": "...", "content_type": "..."}],
#    "patches": [{"doc": "training/<id>", "mask": [...], "fields": {Firestore's typed fields}}]}
#
# sent as: every upload, then every patch. A sent job's files are deleted with
# it. `wildhat training sync` (ai/wildhat/src/wildhat/training.py) sends what's
# left the same way: keep the two in step.


def queue_folder(run_id: str) -> Path:
    return cache_dir() / "training" / run_id / "queue"


def jobs_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("[0-9]*.json")) if folder.is_dir() else []


def send_job(cloud: Cloud, path: Path) -> None:
    job = json.loads(path.read_text())
    for upload in job["uploads"]:
        if "file" in upload:
            cloud.upload(upload["object"], path.parent / upload["file"])
        else:
            cloud.upload(upload["object"], upload["text"].encode(), upload["content_type"])
    for patch in job["patches"]:
        cloud.patch(patch["doc"], patch["mask"], patch["fields"])
    for upload in job["uploads"]:
        if "file" in upload:
            (path.parent / upload["file"]).unlink(missing_ok=True)
    path.unlink()


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
        self.queue = queue_folder(run_id)
        self.queue.mkdir(parents=True, exist_ok=True)
        # Set when resuming: the last resumable checkpoint, downloaded, and its step.
        self.resume_from: Path | None = None
        self.resume_step: int | None = None
        self._rows: list[dict] = []
        self._latest: dict[str, float] = {}
        self._step: int | None = None
        self._sent = 0  # metrics files so far
        self._flushed: float | None = None
        # step -> {"resumable": bool, "pruned": bool}, and the best step
        self._checkpoints: dict[int, dict] = {}
        self._best: int | None = None
        self._jobs = 0
        self._done = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._failing_since: float | None = None
        self._said_failing = False
        self._sender: threading.Thread | None = None

    @property
    def record(self) -> str:
        return f"training/{self.id}"

    # --- Starting ---------------------------------------------------------------

    @classmethod
    def start(cls, model: str, resume: str | None = None, cloud: Cloud | None = None,
              say=None) -> "TrainingRun":
        """Record a new training run of ai/src/training/<model>/, after
        fetching its dataset and finding its [start] checkpoints. With
        `resume`, carry on that run instead (see resume())."""
        cloud, say = cloud or Cloud(), say or print
        if resume:
            run = cls.resume(resume, cloud, say)
            if run.model != model:
                run.finish("stopped")
                raise ValueError(f"Training run {resume} is {run.model}, not {model}")
            return run
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
            "segments": [{**source, "started_at": started, "where": where()}],
            "config": {"path": f"training/{model}/train.toml", "toml": text},
            "dataset": {
                "path": dataset_path.relative_to(SRC.resolve()).as_posix(),
                "toml": dataset_path.read_text(),
                "episodes": {split: dataset.episode_ids(split) for split in SPLITS},
                "samples": {split: len(dataset.samples(split)) for split in SPLITS},
            },
            "parents": parents,
            # The training runs it started from, for finding a run's children.
            "parent_runs": sorted({p["checkpoint"].split("/")[0] for p in parents.values()}),
            "step": None,
            "latest": {},
            "best": None,
            "last_checkpoint": None,
            "checkpoints": 0,
        })
        run = cls(cloud, run_id, model, config, dataset, parents, say)
        run._begin()
        counts = ", ".join(f"{s} {len(dataset.samples(s))}" for s in SPLITS)
        say(f"Training run {run_id}: {model}, by {cloud.handle}. Samples: {counts}.")
        return run

    @classmethod
    def resume(cls, run_id: str, cloud: Cloud | None = None, say=None) -> "TrainingRun":
        """Carry on a run of yours that stopped, from its last resumable
        checkpoint (run.resume_from, run.resume_step). Its train.toml and
        dataset are read again; a warning says if they, or the code, have
        changed since it started."""
        cloud, say = cloud or Cloud(), say or print
        if not REF_RUN.match(run_id):
            raise ValueError(f"{run_id!r} isn't a training run. They look like 20261110-2210-ab12.")
        record = cloud.document(f"training/{run_id}")
        if record is None:
            raise ValueError(f"There's no training run {run_id}")
        if record["trained_by"] != cloud.handle:
            raise ValueError(f"Training run {run_id} is {record['trained_by']}'s, so only they can resume it. "
                             f"Start a new run from its checkpoint instead: in train.toml, [start] then "
                             f"weights = \"{run_id}\"")
        leftover = jobs_in(queue_folder(run_id))
        if leftover:
            say(f"Sending {len(leftover)} upload(s) left from before")
            for path in leftover:
                send_job(cloud, path)
            record = cloud.document(f"training/{run_id}")
        heartbeat = parse_time(record.get("heartbeat_at"))
        if record["status"] == "running" and heartbeat and (now() - heartbeat).total_seconds() < STALE_SECONDS:
            raise ValueError(f"Training run {run_id} is still running (heard from {int((now() - heartbeat).total_seconds())}s "
                             "ago). Stop it first, or wait a few minutes if it just stopped.")

        model = record["model"]
        folder, config, text = read_config(model)
        if text != record["config"]["toml"]:
            say(f"! training/{model}/train.toml has changed since this run started; using it as it is now.")
        dataset_path = SRC / record["dataset"]["path"]
        if not dataset_path.exists():
            raise ValueError(f"This run's dataset, ai/src/{record['dataset']['path']}, isn't there any more")
        if dataset_path.read_text() != record["dataset"]["toml"]:
            say(f"! {record['dataset']['path']} has changed since this run started; using it as it is now.")
        dataset = Dataset.load(dataset_path).fetch(cloud, say)

        checkpoints = {int(c["step"]): c for c in cloud.documents(f"training/{run_id}/checkpoints").values()}
        resumable = [s for s, c in checkpoints.items() if c.get("resumable") and not c.get("pruned")]
        if not resumable:
            raise ValueError(f"Training run {run_id} has no resumable checkpoint to carry on from. "
                             "Save some with run.save(..., resumable=True), or start a new run.")
        step = max(resumable)

        source = source_of(cloud, say)
        last = record["segments"][-1]
        if source.get("source_sha256") != last.get("source_sha256"):
            say("! The code has changed since this run last ran. The record keeps both versions.")
        segments = record["segments"] + [{**source, "started_at": now(), "where": where(), "resumed_at_step": step}]
        cloud.update(f"training/{run_id}", {"status": "running", "heartbeat_at": now(), "finished_at": None,
                                            "error": None, "segments": segments})

        run = cls(cloud, run_id, model, config, dataset, record.get("parents") or {}, say)
        run._checkpoints = {s: {"resumable": bool(c.get("resumable")), "pruned": bool(c.get("pruned"))}
                            for s, c in checkpoints.items()}
        run._best = (record.get("best") or {}).get("step")
        run._sent = len(cloud.objects(f"{run.prefix}metrics/"))
        run._step = record.get("step")
        run._latest = dict(record.get("latest") or {})
        run.resume_step = step
        run.resume_from = download_checkpoint(cloud, f"{run_id}/{step}", checkpoints[step]["file"],
                                              checkpoints[step]["md5"], say)
        run._begin()
        say(f"Resuming training run {run_id} ({model}) from step {step}.")
        return run

    def start_weights(self) -> dict[str, Path]:
        """Each [start] checkpoint's file, downloaded (once) and checked."""
        return {name: download_checkpoint(self.cloud, p["checkpoint"], p["file"], p["md5"], self.say)
                for name, p in self.parents.items()}

    def _begin(self) -> None:
        self._jobs = max([int(p.stem) for p in jobs_in(self.queue)], default=-1) + 1
        (self.queue / ".lock").touch()
        self._sender = threading.Thread(target=self._send, name=f"training-{self.id}", daemon=True)
        self._sender.start()
        self._watch_exit()

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
        """Queue the metrics logged so far to be sent, and print them."""
        with self._lock:
            rows, latest, step = self._rows, dict(self._latest), self._step
            self._rows = []
        self._flushed = time.monotonic()
        if not rows:
            return
        text = "".join(json.dumps(r) + "\n" for r in rows)
        self._enqueue([{"object": f"{self.prefix}metrics/{self._sent:06d}.jsonl", "text": text,
                        "content_type": "application/x-ndjson"}],
                      [(self.record, {"latest": latest, "step": step, "heartbeat_at": now()})])
        self._sent += 1
        self.say(f"{self.model} {self.id} step {step}: " + "  ".join(f"{k} {v:.4g}" for k, v in latest.items()))

    def save(self, path, step: int, *, resumable: bool = False, metrics: dict | None = None,
             best: bool = False) -> str:
        """Upload a checkpoint file (whatever you saved in it) and record it.
        It's copied first, so you can overwrite the file at once; the upload
        happens in the background. `best=True` makes it the run's best, which
        a [start] reference to the run means. Returns its id, "<run>/<step>"."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"There's no checkpoint file {path}")
        step = int(step)
        metrics = {k: float(v) for k, v in (metrics or {}).items()}
        self.flush()
        with self._lock:
            number = self._jobs
        copy = f"{number:06d}-{path.name}"
        shutil.copyfile(path, self.queue / copy)
        name = f"{self.prefix}checkpoints/{step}/{path.name}"
        self._checkpoints[step] = {"resumable": resumable, "pruned": False}
        fields = {"last_checkpoint": step, "checkpoints": len(self._checkpoints), "heartbeat_at": now()}
        # Only the run's best checkpoint says best: a new best clears the old one's.
        was_best = [] if not best or self._best in (None, step) else [
            (f"{self.record}/checkpoints/{self._best}", {"best": False})]
        if best:
            self._best = step
            fields["best"] = {"step": step, "metrics": metrics}
        self._enqueue([{"object": name, "file": copy}], [*was_best,
            (f"{self.record}/checkpoints/{step}", {
                "step": step, "file": name, "size": path.stat().st_size, "md5": md5_of(path),
                "resumable": resumable, "metrics": metrics, "best": best, "saved_at": now(), "tags": [],
                "pruned": False}),
            (self.record, fields),
        ], checkpoint=True)
        return f"{self.id}/{step}"

    def _enqueue(self, uploads: list[dict], patches: list[tuple[str, dict]], checkpoint=False) -> None:
        job = {"uploads": uploads, "checkpoint": checkpoint, "patches": [
            {"doc": doc, "mask": mask, "fields": typed}
            for doc, (mask, typed) in ((d, patch_body(f)) for d, f in patches)]}
        with self._lock:
            number = self._jobs
            self._jobs += 1
        partial = self.queue / f"{number:06d}.json.part"
        partial.write_text(json.dumps(job))
        os.replace(partial, self.queue / f"{number:06d}.json")
        self._wake.set()

    # --- Sending, in the background --------------------------------------------------

    def _send(self) -> None:
        touched = heartbeat = time.monotonic()
        while True:
            if time.monotonic() - touched > TOUCH_SECONDS:
                (self.queue / ".lock").touch()
                touched = time.monotonic()
            jobs = jobs_in(self.queue)
            if not jobs:
                self._failing_since = None
                if self._stopping:
                    return
                if time.monotonic() - heartbeat > HEARTBEAT_SECONDS:
                    try:
                        self.cloud.update(self.record, {"heartbeat_at": now()})
                    except (CloudError, OSError):
                        pass
                    heartbeat = time.monotonic()
                self._wake.wait(TOUCH_SECONDS / 2)
                self._wake.clear()
                continue
            try:
                was_checkpoint = json.loads(jobs[0].read_text()).get("checkpoint")
                send_job(self.cloud, jobs[0])
            except (CloudError, OSError) as e:
                if self._failing_since is None:
                    self._failing_since = time.monotonic()
                if not self._said_failing:
                    self.say(f"Couldn't upload ({e}); trying again, in order, until it works.")
                    self._said_failing = True
                if self._stopping and self._give_up():
                    return
                self._wake.wait(RETRY_SECONDS)
                self._wake.clear()
                continue
            self._failing_since = None
            heartbeat = time.monotonic()
            if was_checkpoint:
                self._prune()

    def _drain(self, timeout: float = 10) -> None:
        """Wait until everything queued so far is sent (for tests)."""
        deadline = time.monotonic() + timeout
        while jobs_in(self.queue) and time.monotonic() < deadline:
            self._wake.set()
            time.sleep(0.01)

    def _give_up(self) -> bool:
        """While finishing: on a laptop, stop after failing for a minute (sync
        sends the rest later); a cloud job keeps trying."""
        return not in_cloud_job() and self._failing_since is not None \
            and time.monotonic() - self._failing_since > GIVE_UP_SECONDS

    def _prune(self) -> None:
        """Delete resumable checkpoints that aren't among the newest few, the
        best, or tagged. Done after uploads, best effort: what isn't deleted
        now is next time."""
        if any(json.loads(p.read_text()).get("checkpoint") for p in jobs_in(self.queue)):
            return  # Not until every checkpoint saved so far is recorded.
        resumable = sorted((s for s, c in self._checkpoints.items() if c["resumable"]), reverse=True)
        for step in resumable[KEEP_RESUMABLE:]:
            if step == self._best or self._checkpoints[step]["pruned"]:
                continue
            try:
                doc = self.cloud.document(f"{self.record}/checkpoints/{step}")
                if doc is None or doc.get("tags"):
                    continue
                try:
                    self.cloud.delete(doc["file"])
                except CloudError as e:
                    if "HTTP 404" not in str(e):
                        raise
                self.cloud.update(f"{self.record}/checkpoints/{step}", {"pruned": True})
                self._checkpoints[step]["pruned"] = True
            except (CloudError, OSError):
                return

    # --- Ending ------------------------------------------------------------------------

    def finish(self, status: str = "finished", error: str | None = None) -> None:
        """Record how it ended, after everything before it is sent."""
        if self._done:
            return
        self._done = True
        self.flush()
        self._enqueue([], [(self.record, {"status": status, "finished_at": now(), "heartbeat_at": now(),
                                          "error": error})])
        waiting = len(jobs_in(self.queue))
        if waiting > 1:
            self.say(f"Sending {waiting} upload(s)...")
        self._stopping = True
        self._wake.set()
        if self._sender:
            self._sender.join()
        left = jobs_in(self.queue)
        if left:
            self.say(f"Couldn't send {len(left)} upload(s) for training run {self.id}, so it isn't recorded as "
                     f"{status} yet. They're kept on this computer: run `wildhat training sync` when you're online.")
            (self.queue / ".lock").unlink(missing_ok=True)
            return
        shutil.rmtree(self.queue, ignore_errors=True)
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
