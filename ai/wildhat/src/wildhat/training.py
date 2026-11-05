"""`wildhat training`: the training database (ai/src/shared/training.py
writes it).

    wildhat training list [--model M] [--mine | --user H] [--limit N]
    wildhat training show RUN
    wildhat training tag CHECKPOINT MODEL:TAG
    wildhat training download CHECKPOINT [--to DIR]
    wildhat training sync

A CHECKPOINT is a training run (its best checkpoint, or its last if none
was best), RUN/STEP, or MODEL:TAG, as in a model's train.toml.

A training run's record is training/<id> in Firestore, its checkpoints'
records training/<id>/checkpoints/<step>, and tags models/<model>
{"tags": {tag: "<run>/<step>"}}. Each tagged checkpoint's record lists its
tags too.
"""

import datetime
import json
import os
import re
import shutil
import time
import urllib.parse
from pathlib import Path

from wildhat.cli import WildhatError
from wildhat.launch import BUCKET, FIRESTORE, STORAGE, request, size
from wildhat.runs import (
    download_object,
    fields,
    inside_source_dir,
    local,
    duration,
    table,
    token,
)

RUN_ID = r"\d{8}-\d{4}-[a-z0-9]{4}"
REF_RUN = re.compile(rf"^({RUN_ID})$")
REF_STEP = re.compile(rf"^({RUN_ID})/(\d+)$")
REF_TAG = re.compile(r"^([a-z_][a-z0-9_]*):([A-Za-z0-9._-]+)$")
# The same as shared/training.py's: a running run is heard from every
# minute, and taken to have stopped after 3 minutes of silence. A queue
# whose lock is newer than LOCK_SECONDS is still being sent by its script.
STALE_SECONDS = 180
LOCK_SECONDS = 120


def cache_dir() -> Path:
    """Where ai/src keeps downloads and training's upload queues (the same as
    shared/datasets.py's)."""
    return Path(os.environ.get("WILDHAT_CACHE") or Path.home() / ".cache" / "wildhat")


# ---------------------------------------------------------------------------
# Reading records
# ---------------------------------------------------------------------------


def get(tok: str, path: str) -> dict | None:
    try:
        return fields(request(tok, "GET", f"{FIRESTORE}/{path}").get("fields", {}))
    except WildhatError as e:
        if "HTTP 404" in str(e):
            return None
        raise


def get_run(tok: str, run_id: str) -> dict:
    if not REF_RUN.match(run_id):
        raise WildhatError(f"{run_id!r} isn't a training run. They look like 20261110-2210-ab12.")
    record = get(tok, f"training/{run_id}")
    if record is None:
        raise WildhatError(f"There's no training run {run_id}.")
    return record


def checkpoints_of(tok: str, run_id: str) -> list[dict]:
    found, page = [], ""
    while True:
        reply = request(tok, "GET", f"{FIRESTORE}/training/{run_id}/checkpoints?pageSize=300&pageToken={page}")
        found += [fields(d["fields"]) for d in reply.get("documents", [])]
        page = reply.get("nextPageToken")
        if not page:
            return sorted(found, key=lambda c: c["step"])


def query(tok: str, where: tuple[str, str, str] | None, limit: int | None, newest_first=True) -> list[tuple[str, dict]]:
    q: dict = {"from": [{"collectionId": "training"}]}
    if newest_first:
        q["orderBy"] = [{"field": {"fieldPath": "created_at"}, "direction": "DESCENDING"}]
    if limit:
        q["limit"] = limit
    if where:
        field, op, value = where
        q["where"] = {"fieldFilter": {"field": {"fieldPath": field}, "op": op, "value": {"stringValue": value}}}
    rows = request(tok, "POST", f"{FIRESTORE}:runQuery", {"structuredQuery": q})
    return [(r["document"]["name"].rsplit("/", 1)[1], fields(r["document"]["fields"])) for r in rows if "document" in r]


def resolve(tok: str, ref: str) -> tuple[str, int, dict, dict]:
    """A checkpoint reference as (run, step, the run's record, the
    checkpoint's record)."""
    tag = REF_TAG.match(ref)
    if tag:
        model, name = tag.groups()
        target = ((get(tok, f"models/{model}") or {}).get("tags") or {}).get(name)
        if not target:
            raise WildhatError(f"No checkpoint is tagged {ref}.")
        return resolve(tok, target)
    whole, one = REF_RUN.match(ref), REF_STEP.match(ref)
    if not (whole or one):
        raise WildhatError(f"{ref!r} isn't a checkpoint. Use a training run (20261110-2210-ab12), one of its "
                           "steps (20261110-2210-ab12/4000) or a tag (fuel_detector:v2).")
    run_id = (whole or one).group(1)
    record = get_run(tok, run_id)
    if one:
        step = int(one.group(2))
    else:
        step = (record.get("best") or {}).get("step")
        step = record.get("last_checkpoint") if step is None else step
        if step is None:
            raise WildhatError(f"Training run {run_id} has no checkpoints yet.")
    checkpoint = get(tok, f"training/{run_id}/checkpoints/{step}")
    if checkpoint is None:
        raise WildhatError(f"Training run {run_id} has no checkpoint at step {step}.")
    return run_id, step, record, checkpoint


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def status_of(record: dict, at: datetime.datetime | None = None) -> str:
    """Its status; "stopped?" for one still recorded as running that hasn't
    been heard from in minutes (a closed laptop, say)."""
    status = record.get("status", "?")
    beat = record.get("heartbeat_at")
    at = at or datetime.datetime.now(datetime.timezone.utc)
    if status == "running" and beat and (at - beat).total_seconds() > STALE_SECONDS:
        return "stopped?"
    return status


def metrics_text(metrics: dict | None) -> str:
    return "  ".join(f"{k} {v:.4g}" if isinstance(v, (int, float)) else f"{k} {v}"
                     for k, v in (metrics or {}).items())


def where_text(where: dict | None) -> str:
    where = where or {}
    if where.get("kind") == "cloud":
        return f"cloud run {where.get('run')}, episode {where.get('episode', 0)}"
    return f"laptop {where.get('host', '?')}"


def code_text(segment: dict) -> str:
    git_state = segment.get("git") or {}
    changes = " with uncommitted changes" if git_state.get("dirty") else ""
    return f"{git_state.get('commit', '?')[:7]} on {git_state.get('branch', '?')}{changes}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(args) -> int:
    tok, me = token()
    user = me if args.mine else args.user
    if args.model:
        found = query(tok, ("model", "EQUAL", args.model), args.limit if not user else None)
        if user:
            found = [(i, r) for i, r in found if r.get("trained_by") == user][:args.limit]
    elif user:
        found = query(tok, ("trained_by", "EQUAL", user), args.limit)
    else:
        found = query(tok, None, args.limit)
    if not found:
        whose = f" by {user}" if user else ""
        print(f"No training runs{' of ' + args.model if args.model else ''}{whose} yet.")
        return 0
    rows = [["RUN", "MODEL", "BY", "STARTED", "STATUS", "STEP", "LATEST"]]
    for run_id, r in found:
        rows.append([run_id, r.get("model", "?"), r.get("trained_by", "?"), local(r.get("created_at")),
                     status_of(r), "" if r.get("step") is None else str(r["step"]), metrics_text(r.get("latest"))])
    width = shutil.get_terminal_size((120, 20)).columns
    used = sum(max(len(row[i]) for row in rows) + 2 for i in range(6))
    table(rows, last_width=max(20, width - used))
    return 0


def cmd_show(args) -> int:
    tok, _ = token()
    record = get_run(tok, args.run_id)
    status = status_of(record)
    took = duration(record.get("created_at"), record.get("finished_at"))
    ended = f" ({local(record['finished_at'])}, after {took})" if record.get("finished_at") else ""
    print(f"Training run {args.run_id}: {record.get('model')}, by {record.get('trained_by')}, "
          f"started {local(record.get('created_at'))}")
    print(f"  status:   {status}{ended}")
    if status == "stopped?":
        print(f"            (last heard from {local(record.get('heartbeat_at'))}; if its laptop was offline, "
              "`wildhat training sync` there sends what's waiting)")
    if record.get("error"):
        print(f"  error:    {record['error']}")
    for i, segment in enumerate(record.get("segments") or []):
        label = "code:" if i == 0 else f"resumed at step {segment.get('resumed_at_step')}:"
        print(f"  {label:<9} {code_text(segment)}, {where_text(segment.get('where') or record.get('where'))}")
    dataset = record.get("dataset") or {}
    samples = ", ".join(f"{k} {v}" for k, v in (dataset.get("samples") or {}).items())
    print(f"  dataset:  {dataset.get('path')} (samples: {samples})")
    for name, parent in (record.get("parents") or {}).items():
        asked = f" (asked as {parent['ref']})" if parent.get("ref") != parent.get("checkpoint") else ""
        print(f"  started from: {name} = {parent.get('checkpoint')}, {parent.get('model')}{asked}")
    children = query(tok, ("parent_runs", "ARRAY_CONTAINS", args.run_id), None, newest_first=False)
    if children:
        print("  used by:  " + ", ".join(f"{i} ({r.get('model')})" for i, r in sorted(children)))
    if record.get("step") is not None:
        print(f"  step:     {record['step']}   {metrics_text(record.get('latest'))}")
    if record.get("best"):
        print(f"  best:     step {record['best']['step']}   {metrics_text(record['best'].get('metrics'))}")
    if record.get("final"):
        print(f"  final:    {metrics_text(record['final'])}")
    if record.get("files"):
        print(f"  files:    {', '.join(f['name'] for f in record['files'])}")
    checkpoints = checkpoints_of(tok, args.run_id)
    if checkpoints:
        print()
        rows = [["CHECKPOINT", "SIZE", "SAVED", "KIND", "METRICS", "TAGS"]]
        for c in checkpoints:
            # The run's best, not every checkpoint that was best when saved.
            best = (record.get("best") or {}).get("step")
            kind = ", ".join(filter(None, ["best" if c["step"] == best else "", "resumable" if c.get("resumable") else "",
                                           "pruned" if c.get("pruned") else ""]))
            rows.append([f"{args.run_id}/{c['step']}", size(int(c.get("size", 0))), local(c.get("saved_at")),
                         kind, metrics_text(c.get("metrics")), " ".join(c.get("tags") or [])])
        table(rows)
    return 0


def field_path(*names: str) -> str:
    """A Firestore field path, each name quoted (tags may have dots)."""
    return ".".join(f"`{n}`" for n in names)


def set_tags(tok: str, path: str, tags: list[str]) -> None:
    request(tok, "PATCH", f"{FIRESTORE}/{path}?updateMask.fieldPaths=tags",
            {"fields": {"tags": {"arrayValue": {"values": [{"stringValue": t} for t in tags]}}}})


def cmd_tag(args) -> int:
    tok, _ = token()
    m = REF_TAG.match(args.tag)
    if not m:
        raise WildhatError(f"{args.tag!r} isn't a tag. Tags look like fuel_detector:v2 (the model, then a name).")
    model, name = m.groups()
    run_id, step, record, checkpoint = resolve(tok, args.checkpoint)
    if record.get("model") != model:
        raise WildhatError(f"{run_id}/{step} is a {record.get('model')} checkpoint, so it can't be {args.tag}.")
    if checkpoint.get("pruned"):
        raise WildhatError(f"{run_id}/{step} was deleted to save space, so it can't be tagged.")
    target = f"{run_id}/{step}"
    before = ((get(tok, f"models/{model}") or {}).get("tags") or {}).get(name)
    if before == target:
        print(f"{args.tag} is {target} already.")
        return 0
    mask = urllib.parse.quote("tags." + field_path(name))
    request(tok, "PATCH", f"{FIRESTORE}/models/{model}?updateMask.fieldPaths={mask}",
            {"fields": {"tags": {"mapValue": {"fields": {name: {"stringValue": target}}}}}})
    set_tags(tok, f"training/{run_id}/checkpoints/{step}", sorted(set(checkpoint.get("tags") or []) | {args.tag}))
    if before:
        old_run, old_step = before.split("/")
        old = get(tok, f"training/{old_run}/checkpoints/{old_step}")
        if old is not None:
            set_tags(tok, f"training/{old_run}/checkpoints/{old_step}",
                     [t for t in old.get("tags") or [] if t != args.tag])
        print(f"{args.tag} is now {target} (it was {before}).")
    else:
        print(f"{args.tag} is now {target}.")
    return 0


def cmd_download(args) -> int:
    tok, _ = token()
    run_id, step, _, checkpoint = resolve(tok, args.checkpoint)
    if checkpoint.get("pruned"):
        raise WildhatError(f"{run_id}/{step} was deleted to save space.")
    dest = Path(args.to or ".")
    if inside_source_dir(dest):
        raise WildhatError("Not downloading into ai/src: the next launch would upload it. "
                           "Run this somewhere else, or use --to.")
    path = dest / Path(checkpoint["file"]).name
    download_object(tok, checkpoint["file"], path)
    print(f"Downloaded {run_id}/{step} ({size(path.stat().st_size)}) to {path}")
    return 0


def send_job(tok: str, path: Path) -> None:
    """Send one queued job as shared/training.py does: every upload, then
    every patch, then delete it and its files."""
    job = json.loads(path.read_text(encoding="utf-8"))
    for upload in job["uploads"]:
        query_text = urllib.parse.urlencode({"uploadType": "media", "name": upload["object"]})
        if "file" in upload:
            data = (path.parent / upload["file"]).read_bytes()
            kind = upload.get("content_type", "application/octet-stream")
        else:
            data, kind = upload["text"].encode(), upload["content_type"]
        request(tok, "POST", f"{STORAGE}/upload/storage/v1/b/{BUCKET}/o?{query_text}", data, kind)
    for patch in job["patches"]:
        mask = "&".join("updateMask.fieldPaths=" + urllib.parse.quote(name) for name in patch["mask"])
        request(tok, "PATCH", f"{FIRESTORE}/{patch['doc']}?{mask}", {"fields": patch["fields"]})
    for upload in job["uploads"]:
        if "file" in upload:
            (path.parent / upload["file"]).unlink(missing_ok=True)
    path.unlink()


def cmd_sync(args) -> int:
    tok, _ = token()
    queues = sorted((cache_dir() / "training").glob("*/queue"))
    sent_any = False
    for queue in queues:
        run_id = queue.parent.name
        jobs = sorted(queue.glob("[0-9]*.json"))
        if not jobs:
            continue
        lock = queue / ".lock"
        if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_SECONDS:
            print(f"{run_id}: its training script is still sending ({len(jobs)} waiting); leaving it to that.")
            continue
        lock.touch()
        try:
            for job in jobs:
                send_job(tok, job)
        finally:
            lock.unlink(missing_ok=True)
        sent_any = True
        print(f"{run_id}: sent {len(jobs)} waiting upload(s).")
        if not any(queue.glob("[0-9]*.json")):
            shutil.rmtree(queue, ignore_errors=True)
    if not sent_any:
        print("Nothing is waiting to be sent.")
    return 0


def add_parsers(sub) -> None:
    training = sub.add_parser("training", help="list, show, tag and download trained models")
    commands = training.add_subparsers(dest="training_command", required=True, metavar="COMMAND")

    listing = commands.add_parser("list", help="the newest training runs")
    listing.add_argument("--model", help="only this model's, e.g. fuel_detector")
    who = listing.add_mutually_exclusive_group()
    who.add_argument("--mine", action="store_true", help="only yours")
    who.add_argument("--user", metavar="HANDLE", help="only this person's, e.g. jane-doe")
    listing.add_argument("--limit", type=int, default=20, help="how many to list (default 20)")
    listing.set_defaults(func=cmd_list)

    show = commands.add_parser("show", help="one training run: its code, dataset, parents and checkpoints")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_show)

    tag = commands.add_parser("tag", help="name a checkpoint, e.g. fuel_detector:v2, for train.toml's [start]")
    tag.add_argument("checkpoint", help="a training run (its best), RUN/STEP, or another tag")
    tag.add_argument("tag", help="MODEL:NAME, e.g. fuel_detector:v2; moves the tag if it's on another checkpoint")
    tag.set_defaults(func=cmd_tag)

    download = commands.add_parser("download", help="download a checkpoint's file")
    download.add_argument("checkpoint", help="a training run (its best), RUN/STEP, or MODEL:TAG")
    download.add_argument("--to", metavar="DIR", help="where to save it (default here)")
    download.set_defaults(func=cmd_download)

    sync = commands.add_parser("sync", help="send uploads a training run on this computer couldn't send")
    sync.set_defaults(func=cmd_sync)
