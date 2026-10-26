"""The base image's entrypoint: fetch a run's source, run its command, and save
what it writes.

`wildhat launch` uploads the source to gs://<bucket>/src/<handle>/<sha256>.tar.gz
and runs the person's Cloud Run job with:

  WILDHAT_SOURCE  gs:// URI of the source tarball
  WILDHAT_OUTPUT  gs:// prefix for the run's outputs, gs://<bucket>/runs/<handle>/<run>/
  args            the command, e.g. `python train.py --epochs 25`

The command runs in the unpacked source, after `pip install -r requirements.txt`
if there is one. Whatever it writes to $WILDHAT_OUTPUT_DIR is uploaded to
WILDHAT_OUTPUT/task-<n>/ when it exits, whether it succeeded or not, and also
when the run is cancelled. Cloud Run's filesystem lives in memory, so those
files count against the job's memory.

Standard library only, so the image needs nothing beyond what the run uses.
"""

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


def say(message: str) -> None:
    print(f"wildhat-run: {message}", flush=True)


def token() -> str:
    """An access token for the job's service account (its launcher's)."""
    req = urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["access_token"]


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


def run(command: list[str], cwd: Path, env: dict) -> int:
    """Run the command, passing on Cloud Run's SIGTERM (a cancel) so it can
    stop cleanly and its outputs still get uploaded."""
    child = subprocess.Popen(command, cwd=cwd, env=env)

    def forward(signum, frame):
        say("stopping (the run was cancelled)")
        child.send_signal(signum)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        code = child.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 128 - code if code < 0 else code


def upload_outputs(output_dir: Path, prefix: str) -> int:
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    for path in files:
        upload(path, prefix + path.relative_to(output_dir).as_posix())
    return len(files)


def main(command: list[str]) -> int:
    if not command:
        raise SystemExit("wildhat-run: no command to run")
    source, output = os.environ["WILDHAT_SOURCE"], os.environ["WILDHAT_OUTPUT"]
    task = os.environ.get("CLOUD_RUN_TASK_INDEX", "0")
    work = Path(os.environ.get("WILDHAT_WORK", "/work"))
    src, output_dir = work / "src", work / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    say(f"fetching {source}")
    fetch_source(source, src)
    env = dict(os.environ, WILDHAT_OUTPUT_DIR=str(output_dir))
    if (src / "requirements.txt").exists():
        say("installing requirements.txt")
        pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
        if subprocess.run(pip, cwd=src, env=env).returncode != 0:
            raise SystemExit("wildhat-run: pip install -r requirements.txt failed")

    say(f"running {' '.join(command)}")
    code = run(command, src, env)
    say(f"exited with {code}")

    prefix = f"{output.rstrip('/')}/task-{task}/"
    try:
        count = upload_outputs(output_dir, prefix)
    except OSError as e:  # urllib's errors are OSErrors
        say(f"couldn't upload outputs: {e}")
        return code or 1
    say(f"uploaded {count} output file(s) to {prefix}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
