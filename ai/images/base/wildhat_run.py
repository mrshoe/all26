"""The base image's entrypoint: fetch a run's source, run its command, and save
what it writes.

`wildhat launch` (ai/wildhat/src/wildhat/launch.py) uploads the source to
gs://<bucket>/src/<handle>/<sha256>.tar.gz and runs the person's Cloud Run job,
one task per episode, with:

  WILDHAT_SOURCE         gs:// URI of the source tarball
  WILDHAT_OUTPUT         gs:// prefix for the run's outputs, gs://<bucket>/runs/<handle>/<run>/
  WILDHAT_RUN_DOC        the run's Firestore document, projects/.../documents/runs/<run>
  WILDHAT_CWD            where in the source to run the command ("" for the top)
  WILDHAT_EPISODE_COUNT  how many episodes the run has
  WILDHAT_BASE_SEED      the run's base seed; each episode's seed is derived from it
  WILDHAT_SEED           (optional) the seed, for a run of one replayed episode
  args                   the command, e.g. `python sim.py --scenario corner`

The command runs in the unpacked source, after `pip install -r requirements.txt`
if the source has one at its top. It gets WILDHAT_EPISODE (0, 1, ...),
WILDHAT_EPISODE_COUNT, WILDHAT_SEED and WILDHAT_OUTPUT_DIR. Whatever it writes
to $WILDHAT_OUTPUT_DIR is uploaded to WILDHAT_OUTPUT/episode-<n>/ when it
exits, whether it succeeded or not, and also when it's stopped by a cancel or
the timeout. Cloud Run's filesystem lives in memory, so those files count
against the job's memory.

The episode's record, runs/<run>/episodes/<n>, gets its status (running,
succeeded, failed or stopped), exit code, times and outputs. Recording is best
effort: a Firestore problem is logged but never fails the episode.

Standard library only, so the image needs nothing beyond what the run uses.
"""

import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

METADATA_TOKEN = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)
STORAGE = "https://storage.googleapis.com"
FIRESTORE = "https://firestore.googleapis.com/v1"


def say(message: str) -> None:
    print(f"wildhat-run: {message}", flush=True)


def token() -> str:
    """An access token for the job's service account (its launcher's)."""
    req = urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["access_token"]


def episode_seed(base_seed: int, episode: int) -> int:
    """The same as episode_seed in ai/wildhat/src/wildhat/launch.py, which
    recorded every episode's seed when it launched the run."""
    digest = hashlib.sha256(f"{base_seed}:{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def record(doc: str, **fields) -> None:
    """Update fields of the episode's Firestore document, as best we can."""
    values = {}
    for name, value in fields.items():
        if isinstance(value, int):
            values[name] = {"integerValue": str(value)}
        elif name.endswith("_at"):
            values[name] = {"timestampValue": value}
        else:
            values[name] = {"stringValue": value}
    mask = "&".join(f"updateMask.fieldPaths={name}" for name in fields)
    req = urllib.request.Request(
        f"{FIRESTORE}/{doc}?{mask}",
        data=json.dumps({"fields": values}).encode(),
        headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"},
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except OSError as e:  # urllib's errors are OSErrors
        say(f"couldn't record {sorted(fields)} in Firestore: {e}")


def split_uri(uri: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(uri)
    if parts.scheme != "gs" or not parts.netloc:
        raise SystemExit(f"wildhat-run: not a gs:// URI: {uri!r}")
    return parts.netloc, parts.path.lstrip("/")


def download(uri: str, path: Path) -> None:
    bucket, name = split_uri(uri)
    url = f"{STORAGE}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(req, timeout=300) as resp, path.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def upload(path: Path, uri: str) -> None:
    bucket, name = split_uri(uri)
    query = urllib.parse.urlencode({"uploadType": "media", "name": name})
    with path.open("rb") as f:
        req = urllib.request.Request(
            f"{STORAGE}/upload/storage/v1/b/{bucket}/o?{query}",
            data=f,
            headers={
                "Authorization": f"Bearer {token()}",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(path.stat().st_size),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600):
            pass


def fetch_source(uri: str, dest: Path) -> None:
    tarball = dest.with_suffix(".tar.gz")
    download(uri, tarball)
    # The tarball is named by its SHA-256, so check it arrived intact.
    expected = Path(split_uri(uri)[1]).name.removesuffix(".tar.gz")
    actual = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"wildhat-run: {uri} is damaged (its SHA-256 is {actual})")
    dest.mkdir(parents=True)
    with tarfile.open(tarball) as tar:
        # "data" refuses absolute paths, ../ and links out of dest.
        tar.extractall(dest, filter="data")
    tarball.unlink()


def run(command: list[str], cwd: Path, env: dict) -> tuple[int, bool]:
    """Run the command, passing on Cloud Run's SIGTERM (a cancel or the
    timeout) so it can stop cleanly and its outputs still get uploaded.
    Returns the exit code, and whether it was stopped."""
    child = subprocess.Popen(command, cwd=cwd, env=env)
    stopped = False

    def forward(signum, frame):
        nonlocal stopped
        stopped = True
        say("stopping (the run was cancelled or timed out)")
        child.send_signal(signum)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        code = child.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return (128 - code if code < 0 else code), stopped


def upload_outputs(output_dir: Path, prefix: str) -> int:
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    for path in files:
        upload(path, prefix + path.relative_to(output_dir).as_posix())
    return len(files)


def prepare(source: str, src: Path, cwd: str, env: dict) -> Path:
    """Fetch and unpack the source and install its requirements. Returns the
    folder to run the command in."""
    fetch_source(source, src)
    run_dir = (src / cwd).resolve()
    if not run_dir.is_relative_to(src.resolve()) or not run_dir.is_dir():
        raise SystemExit(f"wildhat-run: there's no folder {cwd!r} in the source")
    if (src / "requirements.txt").exists():
        say("installing requirements.txt")
        pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
        if subprocess.run(pip, cwd=src, env=env).returncode != 0:
            raise SystemExit("wildhat-run: pip install -r requirements.txt failed")
    return run_dir


def main(command: list[str]) -> int:
    if not command:
        raise SystemExit("wildhat-run: no command to run")
    source, output = os.environ["WILDHAT_SOURCE"], os.environ["WILDHAT_OUTPUT"]
    episode = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    seed = int(os.environ.get("WILDHAT_SEED") or episode_seed(int(os.environ["WILDHAT_BASE_SEED"]), episode))
    doc = f"{os.environ['WILDHAT_RUN_DOC']}/episodes/{episode}"
    cwd = os.environ.get("WILDHAT_CWD", "")
    work = Path(os.environ.get("WILDHAT_WORK", "/work"))
    src, output_dir = work / "src", work / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{output.rstrip('/')}/episode-{episode}/"
    record(doc, status="running", started_at=now())

    say(f"episode {episode}, seed {seed}: fetching {source}")
    env = dict(
        os.environ,
        WILDHAT_OUTPUT_DIR=str(output_dir),
        WILDHAT_EPISODE=str(episode),
        WILDHAT_SEED=str(seed),
    )
    try:
        run_dir = prepare(source, src, cwd, env)
    except BaseException as e:
        # E.g. a damaged source or a bad requirements.txt: the command never ran.
        record(doc, status="failed", error=str(e.code if isinstance(e, SystemExit) else e), finished_at=now())
        raise

    say(f"running {' '.join(command)}")
    code, stopped = run(command, run_dir, env)
    say(f"exited with {code}")

    try:
        count = upload_outputs(output_dir, prefix)
    except OSError as e:  # urllib's errors are OSErrors
        say(f"couldn't upload outputs: {e}")
        code = code or 1
    else:
        say(f"uploaded {count} output file(s) to {prefix}")
    status = "stopped" if stopped else "failed" if code else "succeeded"
    record(doc, status=status, exit_code=code, finished_at=now(), outputs=prefix)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
