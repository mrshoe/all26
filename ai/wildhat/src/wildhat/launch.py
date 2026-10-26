"""`wildhat launch`: run a command in the cloud, as one or more episodes.

    wildhat launch [--episodes N] [--gpu] [--timeout 2h] [--seed S] -- python sim.py

1. Pack ai/src: every file git tracks or would track there (so uncommitted
   edits and new files go too), minus what .gitignore ignores. The tarball is
   reproducible, so the same files always hash the same. The whole team can
   read it, so launch names any files git doesn't track yet.
2. Upload it to gs://<bucket>/src/<handle>/<sha256>.tar.gz, unless it's
   already there. It stays there, so a run's exact code can always be checked.
3. Write the run's record to Firestore: runs/<run-id>, with one document per
   episode in runs/<run-id>/episodes/<n>, each with its seed.
4. Execute the person's Cloud Run job (wh-<handle>, or wh-<handle>-gpu) with
   one task per episode, and return without waiting.

The base image's entrypoint (ai/images/base/wildhat_run.py) does the rest:
it unpacks the source, runs the command with the episode's number and seed,
uploads its outputs to runs/<handle>/<run-id>/episode-<n>/, and updates the
episode's record.

Everything runs as the person's own service account.
"""

import datetime
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from wildhat.cli import (
    IS_WINDOWS,
    PROJECT_ID,
    REGION,
    WildhatError,
    gcloud,
    load_account,
    service_account,
)

BUCKET = f"{PROJECT_ID}-wildhat"
STORAGE = "https://storage.googleapis.com"
FIRESTORE_DB = f"projects/{PROJECT_ID}/databases/(default)"
FIRESTORE = f"https://firestore.googleapis.com/v1/{FIRESTORE_DB}/documents"

MAX_EPISODES = 100
DEFAULT_TIMEOUT = 60 * 60
# The job watchdog cancels anything older (max_job_runtime_hours in
# ai/terraform/guardrails.tf).
MAX_TIMEOUT = 8 * 60 * 60
MAX_SOURCE_BYTES = 100 * 1000 * 1000
# Everything in here runs in the cloud, and nothing else does.
SOURCE_DIR = "ai/src"


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def episode_seed(base_seed: int, episode: int) -> int:
    """Episode `episode`'s seed: a hash of the run's base seed and the episode
    number, as a non-negative 63-bit integer (it fits in Firestore). The base
    image's entrypoint computes the same thing; keep the two in step."""
    digest = hashlib.sha256(f"{base_seed}:{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


# ---------------------------------------------------------------------------
# Packing the source
# ---------------------------------------------------------------------------


def find_root() -> tuple[Path, str]:
    """The folder to pack (ai/src in the person's checkout), and where in it
    they are, as a relative POSIX path ("" at the top)."""
    here = Path.cwd().resolve()
    outside = WildhatError(f"Run wildhat launch from inside {SOURCE_DIR}/ in your all26 checkout.")
    try:
        top = Path(git(here, "rev-parse", "--show-toplevel").strip()).resolve()
    except WildhatError:
        raise outside from None
    root = top / SOURCE_DIR
    if not root.is_dir() or not here.is_relative_to(root):
        raise outside
    cwd = here.relative_to(root).as_posix()
    return root, "" if cwd == "." else cwd


def git(root: Path, *args: str) -> str:
    path = shutil.which("git")
    if not path:
        raise WildhatError("Couldn't find git. Install it: https://git-scm.com/downloads")
    result = subprocess.run([path, "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise WildhatError(f"git {args[0]} failed:\n{result.stderr.strip()}")
    return result.stdout


def source_files(root: Path) -> tuple[list[str], list[str]]:
    """What git tracks or would track under root, as sorted relative POSIX
    paths, and which of those git doesn't track yet."""

    def listed(*args: str) -> set[str]:
        out = git(root, "ls-files", "-z", *args, "--", ".")
        return set(filter(None, out.split("\0")))

    untracked = listed("--others", "--exclude-standard")
    # Not files deleted since the last commit.
    files = sorted(p for p in listed("--cached") | untracked if (root / p).is_file())
    return files, sorted(untracked)


def pack(root: Path, files: list[str]) -> bytes:
    """A .tar.gz of the files that depends only on their names, contents and
    whether they're executable."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in files:
                data = (root / path).read_bytes()
                info = tarfile.TarInfo(path)
                info.size = len(data)
                # Windows calls every file executable, so ignore it there.
                executable = not IS_WINDOWS and os.access(root / path, os.X_OK)
                info.mode = 0o755 if executable else 0o644
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def git_state(root: Path) -> dict:
    commit = git(root, "rev-parse", "HEAD").strip()
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    dirty = bool(git(root, "status", "--porcelain", "--", ".").strip())
    return {"commit": commit, "branch": branch, "dirty": dirty}


# ---------------------------------------------------------------------------
# Google APIs, as the person's service account
# ---------------------------------------------------------------------------


def request(token: str, method: str, url: str, body=None, content_type="application/json"):
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        try:
            message = json.load(e)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = e.reason
        raise WildhatError(f"Google said (HTTP {e.code}): {message}") from None
    except urllib.error.URLError as e:
        raise WildhatError(f"Couldn't reach Google ({e.reason}). Are you online?")


def upload_source(token: str, name: str, data: bytes) -> bool:
    """Upload the tarball unless it's already there. True if it was uploaded."""
    quoted = urllib.parse.quote(name, safe="")
    try:
        request(token, "GET", f"{STORAGE}/storage/v1/b/{BUCKET}/o/{quoted}")
        return False
    except WildhatError as e:
        if "HTTP 404" not in str(e):
            raise
    query = urllib.parse.urlencode({"uploadType": "media", "name": name, "ifGenerationMatch": 0})
    request(token, "POST", f"{STORAGE}/upload/storage/v1/b/{BUCKET}/o?{query}", data, "application/gzip")
    return True


def firestore_value(value) -> dict:
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, datetime.datetime):
        return {"timestampValue": value.isoformat().replace("+00:00", "Z")}
    if isinstance(value, list):
        return {"arrayValue": {"values": [firestore_value(v) for v in value]}}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {k: firestore_value(v) for k, v in value.items()}}}
    if value is None:
        return {"nullValue": None}
    raise TypeError(f"Can't store {value!r} in Firestore")


def create_records(token: str, run_id: str, run: dict, episodes: list[dict]) -> None:
    """Create the run's document and its episodes' in one commit, failing if
    the run already exists."""
    doc = f"{FIRESTORE_DB}/documents/runs/{run_id}"
    writes = [{"update": {"name": doc, "fields": firestore_value(run)["mapValue"]["fields"]},
               "currentDocument": {"exists": False}}]
    for episode in episodes:
        writes.append({
            "update": {"name": f"{doc}/episodes/{episode['index']}",
                       "fields": firestore_value(episode)["mapValue"]["fields"]},
            "currentDocument": {"exists": False},
        })
    request(token, "POST", f"{FIRESTORE}:commit", {"writes": writes})


def update_record(token: str, run_id: str, fields: dict) -> None:
    mask = "&".join(f"updateMask.fieldPaths={k}" for k in fields)
    request(token, "PATCH", f"{FIRESTORE}/runs/{run_id}?{mask}",
            {"fields": firestore_value(fields)["mapValue"]["fields"]})


def delete_records(token: str, run_id: str, episodes: int) -> None:
    for path in [*(f"runs/{run_id}/episodes/{i}" for i in range(episodes)), f"runs/{run_id}"]:
        try:
            request(token, "DELETE", f"{FIRESTORE}/{path}")
        except WildhatError:
            pass


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> int:
    """Seconds in "90m", "2h", "45s" or plain seconds."""
    m = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", text.lower())
    if not m:
        raise WildhatError(f"Can't read the timeout {text!r}. Try 90m or 2h.")
    return int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def new_run_id() -> str:
    alphabet = "abcdefghijkmnpqrstuvwxyz23456789"  # No l/1 or o/0 to misread.
    suffix = "".join(secrets.choice(alphabet) for _ in range(4))
    return datetime.datetime.now().strftime("%Y%m%d-%H%M-") + suffix


def size(n: int) -> str:
    if n >= 100_000:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e3:.0f} KB" if n >= 1000 else f"{n} B"


def cmd_launch(args) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise WildhatError("Say what to run after --, e.g. wildhat launch -- python train.py")
    if not 1 <= args.episodes <= MAX_EPISODES:
        raise WildhatError(f"--episodes must be between 1 and {MAX_EPISODES}.")
    timeout = parse_duration(args.timeout)
    if not 1 <= timeout <= MAX_TIMEOUT:
        raise WildhatError(f"--timeout can be at most {MAX_TIMEOUT // 3600}h.")
    if args.seed is not None and args.episodes != 1:
        raise WildhatError("--seed replays a single episode, so it needs --episodes 1.")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        raise WildhatError("--seed must be between 0 and 2^63 - 1.")

    account = load_account()
    handle = account["handle"]
    sa = service_account(handle)
    root, cwd = find_root()
    files, untracked = source_files(root)
    data = pack(root, files)
    if len(data) > MAX_SOURCE_BYTES:
        raise WildhatError(
            f"{SOURCE_DIR} packs to {size(len(data))}, over the {size(MAX_SOURCE_BYTES)} "
            "limit. Put data in the bucket instead, or add big folders to .gitignore."
        )
    sha = hashlib.sha256(data).hexdigest()
    source = f"src/{handle}/{sha}.tar.gz"
    state = git_state(root)

    token = gcloud("auth", "print-access-token", f"--account={sa}").stdout.strip()
    uploaded = upload_source(token, source, data)
    changes = ", with uncommitted changes" if state["dirty"] else ""
    print(
        f"Packed {len(files)} files ({size(len(data))}) from {SOURCE_DIR}/ at "
        f"{state['commit'][:7]}{changes}" + ("." if uploaded else "; already uploaded.")
    )
    if untracked:
        # The whole team can read what's uploaded, so show what might be a
        # private note rather than code.
        print(f"Included {len(untracked)} file(s) git doesn't track yet: " + ", ".join(untracked))

    run_id = new_run_id()
    job = f"wh-{handle}" + ("-gpu" if args.gpu else "")
    output = f"gs://{BUCKET}/runs/{handle}/{run_id}/"
    base_seed = secrets.randbits(63)
    seeds = [args.seed] if args.seed is not None else [
        episode_seed(base_seed, i) for i in range(args.episodes)
    ]
    run = {
        "launched_by": handle,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "command": command,
        "cwd": cwd,
        "source": f"gs://{BUCKET}/{source}",
        "source_sha256": sha,
        "git": state,
        "episodes": args.episodes,
        "base_seed": base_seed,
        "gpu": args.gpu,
        "timeout_seconds": timeout,
        "job": job,
        "execution": None,
        "outputs": output,
    }
    episodes = [{"index": i, "seed": seed, "status": "pending"} for i, seed in enumerate(seeds)]
    create_records(token, run_id, run, episodes)

    env = {
        "WILDHAT_SOURCE": run["source"],
        "WILDHAT_OUTPUT": output,
        "WILDHAT_RUN_DOC": f"{FIRESTORE_DB}/documents/runs/{run_id}",
        "WILDHAT_CWD": cwd,
        "WILDHAT_EPISODE_COUNT": str(args.episodes),
        "WILDHAT_BASE_SEED": str(base_seed),
    }
    if args.seed is not None:
        env["WILDHAT_SEED"] = str(args.seed)
    # The command and environment go in a flags file, not on gcloud's command
    # line: on Windows gcloud is a .cmd script, and cmd.exe would mangle
    # | & ^ % in them.
    with tempfile.TemporaryDirectory() as tmp:
        flags = Path(tmp) / "flags.json"
        flags.write_text(json.dumps({"--update-env-vars": env, "--args": command}), encoding="utf-8")
        result = gcloud(
            "run", "jobs", "execute", job,
            f"--region={REGION}",
            f"--account={sa}",
            "--async",
            f"--tasks={args.episodes}",
            f"--task-timeout={timeout}s",
            f"--flags-file={flags}",
            "--format=json",
            check=False,
        )
    if result.returncode != 0:
        delete_records(token, run_id, args.episodes)
        if args.gpu and "NOT_FOUND" in result.stderr:
            raise WildhatError("GPU jobs aren't set up yet (the project's GPU quota is 0). Ask a mentor.")
        raise WildhatError(f"Couldn't start the run:\n{result.stderr.strip()}")
    execution = json.loads(result.stdout)["metadata"]["name"]
    update_record(token, run_id, {"execution": execution})

    where = "a GPU" if args.gpu else "CPU"
    count = f"{args.episodes} episodes" if args.episodes > 1 else "1 episode"
    print(f"Launched run {run_id}: {count} on {where}, up to {timeout // 60} min each.")
    print(f"Outputs: {output}")
    return 0


def add_parsers(sub) -> None:
    launch = sub.add_parser(
        "launch",
        help="run a command in the cloud",
        usage="wildhat launch [--episodes N] [--gpu] [--timeout 1h] [--seed S] -- COMMAND ...",
    )
    launch.add_argument(
        "--episodes", type=int, default=1,
        help=f"how many copies to run, each with its own seed (default 1, at most {MAX_EPISODES})",
    )
    launch.add_argument("--gpu", action="store_true", help="give each episode an NVIDIA L4")
    launch.add_argument(
        "--timeout", default="1h",
        help=f"how long each episode may run, e.g. 90m (default 1h, at most {MAX_TIMEOUT // 3600}h)",
    )
    launch.add_argument("--seed", type=int, help="the seed for a single episode, to replay one")
    launch.add_argument("command", nargs="...", help="what to run, after --")
    launch.set_defaults(func=cmd_launch)
