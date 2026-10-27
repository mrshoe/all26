"""Following runs: `wildhat runs`, `logs`, `cancel` and `download`.

A run's record is in Firestore (runs/<id>, written by launch), its episodes'
records under it (updated by the base image's entrypoint as they go), its
progress in the Cloud Run execution the record names, its output in Cloud
Logging, and its files in the bucket. Status is worked out from the execution
and the episode records each time, never stored: an episode cancelled before
it started never updates its record, for example.

Everyone can see everyone's runs. Only `cancel` is limited to your own, which
Cloud Run enforces anyway: you may only run and cancel your own job.
"""

import concurrent.futures
import datetime
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from wildhat.admin import confirm
from wildhat.cli import (
    PROJECT_ID,
    REGION,
    WildhatError,
    gcloud,
    load_account,
    service_account,
)
from wildhat.launch import (
    BUCKET,
    FIRESTORE,
    SOURCE_DIR,
    STORAGE,
    git,
    request,
    size,
)

RUN_API = f"https://run.googleapis.com/v2/projects/{PROJECT_ID}/locations/{REGION}/jobs"
LOGGING = "https://logging.googleapis.com/v2/entries:list"
RUN_ID = re.compile(r"^\d{8}-\d{4}-[a-z0-9]{4}$")
FOLLOW_SECONDS = 5
# Cloud Logging can take a while to make a line searchable, so -f keeps
# reading for this long after the run ends.
LOG_DELAY_SECONDS = 30
# It doesn't make lines searchable in order, either, so each read looks back
# this far before the newest line so far.
LOG_LOOKBACK_SECONDS = 60


def token() -> tuple[str, str]:
    """An access token for the person's service account, and their handle."""
    handle = load_account()["handle"]
    sa = service_account(handle)
    return gcloud("auth", "print-access-token", f"--account={sa}").stdout.strip(), handle


# ---------------------------------------------------------------------------
# Reading records
# ---------------------------------------------------------------------------


def plain(value: dict):
    """A Firestore typed value ({"integerValue": "3"}) as plain Python."""
    (kind, v), = value.items()
    if kind == "integerValue":
        return int(v)
    if kind == "timestampValue":
        return parse_time(v)
    if kind == "arrayValue":
        return [plain(x) for x in v.get("values", [])]
    if kind == "mapValue":
        return fields(v.get("fields", {}))
    if kind == "nullValue":
        return None
    return v


def parse_time(stamp: str) -> datetime.datetime:
    """An RFC 3339 time like 2026-10-26T06:46:28.664906Z (Python 3.10's
    fromisoformat takes neither the Z nor more than 6 fraction digits)."""
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)", stamp)
    if not m:
        raise WildhatError(f"Can't read the time {stamp!r}")
    when = datetime.datetime.fromisoformat(m.group(1) + ("+00:00" if m.group(3) == "Z" else m.group(3)))
    return when.replace(microsecond=int((m.group(2) or "0")[:6].ljust(6, "0")))


def fields(doc_fields: dict) -> dict:
    return {k: plain(v) for k, v in doc_fields.items()}


def find_runs(tok: str, user: str | None, limit: int) -> list[tuple[str, dict]]:
    """The newest runs, everyone's or one person's, as (id, record)."""
    query = {
        "from": [{"collectionId": "runs"}],
        "orderBy": [{"field": {"fieldPath": "created_at"}, "direction": "DESCENDING"}],
        "limit": limit,
    }
    if user:
        query["where"] = {"fieldFilter": {
            "field": {"fieldPath": "launched_by"}, "op": "EQUAL", "value": {"stringValue": user},
        }}
    rows = request(tok, "POST", f"{FIRESTORE}:runQuery", {"structuredQuery": query})
    return [
        (row["document"]["name"].rsplit("/", 1)[1], fields(row["document"]["fields"]))
        for row in rows if "document" in row
    ]


def get_run(tok: str, run_id: str) -> dict:
    if not RUN_ID.match(run_id):
        raise WildhatError(f"{run_id!r} isn't a run ID. They look like 20261026-1430-ab12.")
    try:
        return fields(request(tok, "GET", f"{FIRESTORE}/runs/{run_id}")["fields"])
    except WildhatError as e:
        if "HTTP 404" in str(e):
            raise WildhatError(f"There's no run {run_id}.") from None
        raise


def get_episodes(tok: str, run_id: str) -> list[dict]:
    episodes, page = [], ""
    while True:
        reply = request(tok, "GET", f"{FIRESTORE}/runs/{run_id}/episodes?pageSize=300&pageToken={page}")
        episodes += [fields(d["fields"]) for d in reply.get("documents", [])]
        page = reply.get("nextPageToken")
        if not page:
            return sorted(episodes, key=lambda e: e["index"])


def execution_url(run: dict) -> str | None:
    if not run.get("execution"):
        return None
    return f"{RUN_API}/{run['job']}/executions/{run['execution']}"


def get_execution(tok: str, run: dict) -> dict | None:
    url = execution_url(run)
    if not url:
        return None
    try:
        return request(tok, "GET", url)
    except WildhatError as e:
        if "HTTP 404" in str(e):
            return None
        raise


# ---------------------------------------------------------------------------
# Status, worked out from the execution and the episode records
# ---------------------------------------------------------------------------


def run_status(run: dict, execution: dict | None) -> str:
    total = run["episodes"]
    if execution is None:
        return "not started" if not run.get("execution") else "unknown"
    done = {k: execution.get(f"{k}Count", 0) for k in ("succeeded", "failed", "cancelled")}
    if not execution.get("completionTime"):
        finished = sum(done.values())
        if not finished and not execution.get("runningCount"):
            return "starting"
        return f"running ({finished} of {total} done)" if total > 1 else "running"
    if done["cancelled"]:
        return "cancelled"
    if done["failed"]:
        return f"failed ({done['failed']} of {total})" if total > 1 else "failed"
    return "succeeded"


def episode_status(episode: dict, execution: dict | None) -> str:
    """The episode's recorded status, unless the run has ended without it
    ever reporting (cancelled before it started, say)."""
    status = episode["status"]
    ended = execution is None or execution.get("completionTime")
    if ended and status == "pending":
        return "not started"
    if ended and status == "running":
        return "unknown"  # It stopped without saying how.
    return status


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def local(when: datetime.datetime | None) -> str:
    return when.astimezone().strftime("%b %d %H:%M") if when else ""


def duration(start, end) -> str:
    if not start or not end:
        return ""
    seconds = int((end - start).total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def table(rows: list[list[str]], last_width: int | None = None) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths)]
        line = "  ".join(cells).rstrip()
        if last_width:
            prefix = "  ".join(cells[:-1])
            if len(row[-1]) > last_width:
                line = f"{prefix}  {row[-1][:last_width - 1]}…"
        print(line)


def command_line(run: dict) -> str:
    return " ".join(run["command"])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_runs(args) -> int:
    tok, me = token()
    if args.run_id:
        return show_run(tok, args.run_id)
    user = me if args.mine else args.user
    found = find_runs(tok, user, args.limit)
    if not found:
        print("No runs yet." if not user else f"No runs by {user} yet.")
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        executions = list(pool.map(lambda r: get_execution(tok, r[1]), found))
    rows = [["RUN", "BY", "STARTED", "EPISODES", "STATUS", "COMMAND"]]
    for (run_id, run), execution in zip(found, executions):
        rows.append([
            run_id,
            run["launched_by"],
            local(run.get("created_at")),
            str(run["episodes"]) + (" GPU" if run.get("gpu") else ""),
            run_status(run, execution),
            command_line(run),
        ])
    width = shutil.get_terminal_size((120, 20)).columns
    used = sum(max(len(r[i]) for r in rows) + 2 for i in range(5))
    table(rows, last_width=max(20, width - used))
    return 0


def show_run(tok: str, run_id: str) -> int:
    run = get_run(tok, run_id)
    execution = get_execution(tok, run)
    episodes = get_episodes(tok, run_id)
    git_state = run.get("git", {})
    changes = " with uncommitted changes" if git_state.get("dirty") else ""
    print(f"Run {run_id}, launched by {run['launched_by']} on {local(run.get('created_at'))}")
    print(f"  status:   {run_status(run, execution)}")
    folder = "/".join(filter(None, [SOURCE_DIR, run.get("cwd")]))
    print(f"  command:  {command_line(run)}   (in {folder})")
    print(f"  code:     {git_state.get('commit', '')[:7]} on {git_state.get('branch', '?')}{changes}")
    where = "an L4 GPU" if run.get("gpu") else "CPU"
    print(f"  on:       {where}, up to {duration_text(run['timeout_seconds'])} per episode")
    print()
    rows = [["EPISODE", "SEED", "STATUS", "EXIT", "TIME"]]
    for e in episodes:
        rows.append([
            str(e["index"]),
            str(e["seed"]),
            episode_status(e, execution),
            "" if e.get("exit_code") is None else str(e["exit_code"]),
            duration(e.get("started_at"), e.get("finished_at")),
        ])
    table(rows)
    return 0


def duration_text(seconds: int) -> str:
    return f"{seconds // 3600}h" if seconds % 3600 == 0 else f"{seconds // 60} min"


def log_entries(tok: str, run: dict, episode: int | None, after: str | None) -> list[dict]:
    """The run's log entries, oldest first, after the timestamp `after`."""
    flt = [
        'resource.type="cloud_run_job"',
        # The run's stdout, stderr and Cloud Run's notes about it ("Terminating
        # task because..."), not audit logs.
        'logName:"logs/run.googleapis.com"',
        f'labels."run.googleapis.com/execution_name"="{run["execution"]}"',
        # Cloud Logging searches only the last day unless told otherwise.
        f'timestamp>="{(after or run["created_at"].isoformat().replace("+00:00", "Z"))}"',
    ]
    if episode is not None:
        flt.append(f'labels."run.googleapis.com/task_index"="{episode}"')
    entries, page = [], None
    while True:
        body = {
            "resourceNames": [f"projects/{PROJECT_ID}"],
            "filter": " AND ".join(flt),
            "orderBy": "timestamp asc",
            "pageSize": 1000,
        }
        if page:
            body["pageToken"] = page
        reply = request(tok, "POST", LOGGING, body)
        entries += reply.get("entries", [])
        page = reply.get("nextPageToken")
        if not page:
            return entries


def entry_text(entry: dict) -> str:
    if "textPayload" in entry:
        return entry["textPayload"]
    payload = entry.get("jsonPayload", {})
    return payload.get("message") or str(payload)


def cmd_logs(args) -> int:
    tok, _ = token()
    run = get_run(tok, args.run_id)
    if not run.get("execution"):
        raise WildhatError(f"Run {args.run_id} never started, so it has no logs.")
    if args.episode is not None and not 0 <= args.episode < run["episodes"]:
        raise WildhatError(f"Run {args.run_id} has episodes 0 to {run['episodes'] - 1}.")
    seen: set[str] = set()
    newest = None
    while True:
        # Ask whether it has ended before reading, so nothing logged at the
        # very end is missed, and keep reading a while after, for lines
        # Cloud Logging is still taking in.
        ended = True
        if args.follow:
            execution = get_execution(tok, run) or {}
            finished = execution.get("completionTime")
            ended = bool(finished) and (
                datetime.datetime.now(datetime.timezone.utc) - parse_time(finished)
            ).total_seconds() > LOG_DELAY_SECONDS
        # Not just lines after the newest so far: one written earlier can show
        # up later. Lines already printed are skipped by their insertId.
        after = None
        if newest:
            after = (newest - datetime.timedelta(seconds=LOG_LOOKBACK_SECONDS)).isoformat().replace("+00:00", "Z")
        for entry in log_entries(tok, run, args.episode, after):
            if entry["insertId"] in seen:
                continue
            seen.add(entry["insertId"])
            newest = max(filter(None, [newest, parse_time(entry["timestamp"])]))
            text = entry_text(entry).rstrip("\n")
            if args.episode is None and run["episodes"] > 1:
                index = entry.get("labels", {}).get("run.googleapis.com/task_index", "?")
                text = f"[{index}] {text}"
            print(text, flush=True)
        if ended:
            break
        time.sleep(FOLLOW_SECONDS)
    if not seen:
        print("No logs yet.")
    return 0


def cmd_cancel(args) -> int:
    tok, me = token()
    run = get_run(tok, args.run_id)
    if run["launched_by"] != me:
        raise WildhatError(
            f"Run {args.run_id} is {run['launched_by']}'s; you can only cancel your own. Ask a mentor."
        )
    execution = get_execution(tok, run)
    if execution is None or execution.get("completionTime"):
        print(f"Run {args.run_id} has already ended ({run_status(run, execution)}).")
        return 0
    status = run_status(run, execution)
    if not args.yes and not confirm(f"Cancel run {args.run_id}: {command_line(run)} [{status}]?"):
        print("Not cancelled.")
        return 1
    request(tok, "POST", f"{execution_url(run)}:cancel", {})
    print(f"Cancelling run {args.run_id}. Running episodes get a few seconds to save their outputs.")
    return 0


def inside_source_dir(path: Path) -> bool:
    """Whether path is in ai/src, where the next launch would upload it."""
    try:
        top = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip()).resolve()
    except WildhatError:
        return False
    return path.resolve().is_relative_to(top / SOURCE_DIR)


def list_objects(tok: str, prefix: str) -> list[dict]:
    items, page = [], ""
    while True:
        query = urllib.parse.urlencode({"prefix": prefix, "pageToken": page, "fields": "items(name,size),nextPageToken"})
        reply = request(tok, "GET", f"{STORAGE}/storage/v1/b/{BUCKET}/o?{query}")
        items += reply.get("items", [])
        page = reply.get("nextPageToken")
        if not page:
            return items


def download_object(tok: str, name: str, path: Path) -> None:
    """Stream one object to path (outputs can be big checkpoints)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(name, safe="")
    req = urllib.request.Request(
        f"{STORAGE}/storage/v1/b/{BUCKET}/o/{quoted}?alt=media",
        headers={"Authorization": f"Bearer {tok}"},
    )
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, partial.open("wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)
    except urllib.error.HTTPError as e:
        with e:  # Close the error's body, or Python warns later.
            raise WildhatError(f"Couldn't download {name} (HTTP {e.code}).") from None
    except urllib.error.URLError as e:
        raise WildhatError(f"Couldn't reach Google ({e.reason}). Are you online?")
    os.replace(partial, path)


def unpack(tarball: Path, dest: Path) -> None:
    with tarfile.open(tarball) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:  # Python 3.10 before 3.10.12
            for member in tar.getmembers():
                target = (dest / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(dest.resolve()):
                    raise WildhatError(f"Refusing to unpack {member.name!r} from {tarball}")
            tar.extractall(dest)


def cmd_download(args) -> int:
    tok, _ = token()
    run = get_run(tok, args.run_id)
    dest = Path(args.to or args.run_id)
    if inside_source_dir(dest):
        raise WildhatError(
            f"Not downloading into {SOURCE_DIR}: the next launch would upload it all again. "
            "Run this somewhere else, or use --to."
        )
    if args.source:
        name = run["source"].removeprefix(f"gs://{BUCKET}/")
        tarball = dest / "source.tar.gz"
        download_object(tok, name, tarball)
        unpack(tarball, dest / "source")
        tarball.unlink()
        print(f"Downloaded the code run {args.run_id} ran to {dest / 'source'}")
        return 0

    prefix = run["outputs"].removeprefix(f"gs://{BUCKET}/")
    if args.episode is not None:
        prefix += f"episode-{args.episode}/"
    objects = list_objects(tok, prefix)
    if not objects:
        print(f"Run {args.run_id} has no outputs" + (f" for episode {args.episode}." if args.episode is not None else "."))
        return 0
    base = run["outputs"].removeprefix(f"gs://{BUCKET}/")
    total = 0
    for obj in objects:
        relative = obj["name"].removeprefix(base)
        target = (dest / relative).resolve()
        if not target.is_relative_to(dest.resolve()):
            raise WildhatError(f"Refusing to write {obj['name']} outside {dest}")
        download_object(tok, obj["name"], dest / relative)
        total += int(obj.get("size", 0))
    print(f"Downloaded {len(objects)} file(s) ({size(total)}) to {dest}{os.sep}")
    return 0


def add_parsers(sub) -> None:
    runs = sub.add_parser("runs", help="list runs, or show one")
    runs.add_argument("run_id", nargs="?", help="show this run in detail")
    who = runs.add_mutually_exclusive_group()
    who.add_argument("--mine", action="store_true", help="only your runs")
    who.add_argument("--user", metavar="HANDLE", help="only this person's runs, e.g. jane-doe")
    runs.add_argument("--limit", type=int, default=20, help="how many runs to list (default 20)")
    runs.set_defaults(func=cmd_runs)

    logs = sub.add_parser("logs", help="print a run's output")
    logs.add_argument("run_id")
    logs.add_argument("--episode", type=int, help="only this episode's output")
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing until the run ends")
    logs.set_defaults(func=cmd_logs)

    cancel = sub.add_parser("cancel", help="cancel one of your runs")
    cancel.add_argument("run_id")
    cancel.add_argument("--yes", action="store_true", help="don't ask first")
    cancel.set_defaults(func=cmd_cancel)

    download = sub.add_parser("download", help="download a run's outputs (or its code)")
    download.add_argument("run_id")
    download.add_argument("--episode", type=int, help="only this episode's outputs")
    download.add_argument("--source", action="store_true", help="the code it ran, instead of its outputs")
    download.add_argument("--to", metavar="DIR", help="where to save (default ./RUN_ID)")
    download.set_defaults(func=cmd_download)
