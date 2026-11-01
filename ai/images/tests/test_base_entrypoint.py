"""Tests for the base image's entrypoint, images/base/wildhat_run.py.

They run the real file against an in-memory Cloud Storage and metadata server
(urlopen is replaced), with real commands in a temp folder standing in for
the container's /work. Run from ai/images:

    python -m unittest discover -s tests
"""

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path
from unittest import mock

ENTRYPOINT = Path(__file__).resolve().parents[1] / "base" / "wildhat_run.py"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("wildhat_run", ENTRYPOINT)
wildhat_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wildhat_run)

BUCKET = "vernal-mantra-512116-d0-wildhat"
RUN = "20261026-1430-ab12"
OUTPUT = f"gs://{BUCKET}/runs/jane-doe/{RUN}/"
RUN_DOC = f"projects/vernal-mantra-512116-d0/databases/(default)/documents/runs/{RUN}"
EPISODE_DOC = f"{RUN_DOC}/episodes/3"
BASE_SEED = 1234567890123


def wheel(name: str, version: str, code: str) -> bytes:
    """A tiny one-module wheel, which pip installs without going online."""
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}.py": code,
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{dist}/RECORD"] = "".join(f"{path},,\n" for path in [*files, f"{dist}/RECORD"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, text in files.items():
            zf.writestr(path, text)
    return buf.getvalue()


class FakeCloud:
    """The metadata server and Cloud Storage, as the job's account sees them."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        # Firestore: each PATCH to a document, as {field: value}, in order.
        self.updates: dict[str, list[dict]] = {}
        self.firestore_down = False

    def urlopen(self, req, timeout=None):
        url = urllib.parse.urlsplit(req.full_url)
        if url.hostname == "metadata.google.internal":
            assert req.get_header("Metadata-flavor") == "Google"
            return io.BytesIO(json.dumps({"access_token": "ya29.job", "expires_in": 3599}).encode())
        assert req.get_header("Authorization") == "Bearer ya29.job"
        if url.hostname == "firestore.googleapis.com":
            return self.firestore(req, url)
        assert url.hostname == "storage.googleapis.com", req.full_url
        query = dict(urllib.parse.parse_qsl(url.query))
        path = urllib.parse.unquote(url.path)
        if path.startswith("/storage/v1/b/") and query.get("alt") == "media":
            bucket, name = path[len("/storage/v1/b/"):].split("/o/", 1)
            if (bucket, name) not in self.objects:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, io.BytesIO(b"{}"))
            return io.BytesIO(self.objects[bucket, name])
        if path.startswith("/upload/storage/v1/b/") and req.get_method() == "POST":
            assert query["uploadType"] == "media"
            bucket = path[len("/upload/storage/v1/b/"):].split("/o", 1)[0]
            body = req.data.read()
            assert len(body) == int(req.get_header("Content-length"))
            self.objects[bucket, query["name"]] = body
            return io.BytesIO(b"{}")
        raise AssertionError(f"Unexpected request: {req.get_method()} {req.full_url}")

    def firestore(self, req, url):
        if self.firestore_down:
            raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", None, io.BytesIO(b"{}"))
        assert req.get_method() == "PATCH"
        doc = urllib.parse.unquote(url.path).removeprefix("/v1/")
        mask = urllib.parse.parse_qs(url.query)["updateMask.fieldPaths"]
        fields = json.loads(req.data)["fields"]
        assert sorted(mask) == sorted(fields), (mask, fields)
        self.updates.setdefault(doc, []).append({k: next(iter(v.values())) for k, v in fields.items()})
        return io.BytesIO(b"{}")

    def episode(self) -> dict:
        """The episode's record, after all its updates."""
        merged = {}
        for update in self.updates.get(EPISODE_DOC, []):
            merged.update(update)
        return merged

    def outputs(self) -> dict[str, bytes]:
        prefix = OUTPUT[len(f"gs://{BUCKET}/"):]
        return {n[len(prefix):]: d for (b, n), d in self.objects.items() if n.startswith(prefix)}


class EntrypointTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="wildhat run ")
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name) / "work"
        self.cloud = FakeCloud()
        patcher = mock.patch("urllib.request.urlopen", self.cloud.urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.env = {
            "WILDHAT_OUTPUT": OUTPUT,
            "WILDHAT_RUN_DOC": RUN_DOC,
            "WILDHAT_CWD": "",
            "WILDHAT_EPISODE_COUNT": "5",
            "WILDHAT_BASE_SEED": str(BASE_SEED),
            "WILDHAT_WORK": str(self.work),
            "CLOUD_RUN_TASK_INDEX": "3",
        }

    def upload_source(self, files: dict[str, str | bytes], *, damage: bool = False) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, content in files.items():
                data = content if isinstance(content, bytes) else content.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        data = buf.getvalue()
        name = f"src/jane-doe/{hashlib.sha256(data).hexdigest()}.tar.gz"
        self.cloud.objects[BUCKET, name] = data + (b"x" if damage else b"")
        self.env["WILDHAT_SOURCE"] = f"gs://{BUCKET}/{name}"

    def run_entrypoint(self, *command: str) -> int:
        with mock.patch.dict(os.environ, self.env):
            try:
                return wildhat_run.main(list(command))
            except SystemExit as e:
                return e.code

    def test_runs_the_command_in_the_source_and_uploads_its_outputs(self):
        self.upload_source({
            "train.py": (
                "import os, pathlib, sys\n"
                "out = pathlib.Path(os.environ['WILDHAT_OUTPUT_DIR'])\n"
                "(out / 'weights').mkdir()\n"
                "(out / 'weights' / 'best.pt').write_text('weights ' + ' '.join(sys.argv[1:]))\n"
                "(out / 'results.csv').write_text(open('data/labels.txt').read())\n"
            ),
            "data/labels.txt": "ball\n",
        })

        code = self.run_entrypoint(sys.executable, "train.py", "--epochs", "25")

        self.assertEqual(code, 0)
        self.assertEqual(self.cloud.outputs(), {
            "episode-3/weights/best.pt": b"weights --epochs 25",
            "episode-3/results.csv": b"ball\n",
        })

    def test_a_failing_command_still_uploads_its_outputs_and_fails_the_task(self):
        self.upload_source({"crash.py": (
            "import os, pathlib\n"
            "pathlib.Path(os.environ['WILDHAT_OUTPUT_DIR'], 'log.txt').write_text('got this far')\n"
            "raise SystemExit(7)\n"
        )})

        code = self.run_entrypoint(sys.executable, "crash.py")

        self.assertEqual(code, 7)
        self.assertEqual(self.cloud.outputs(), {"episode-3/log.txt": b"got this far"})

    @unittest.skipIf(os.name == "nt", "the image is Linux; Windows has no SIGTERM to pass on")
    def test_a_cancel_reaches_the_command_and_its_outputs_are_kept(self):
        # Cloud Run cancels a task by sending SIGTERM to the entrypoint. This
        # command plays both parts: it sends that SIGTERM, then waits for the
        # entrypoint to pass it on.
        self.upload_source({"long.py": (
            "import os, pathlib, signal, time\n"
            "out = pathlib.Path(os.environ['WILDHAT_OUTPUT_DIR'])\n"
            "def stop(signum, frame):\n"
            "    (out / 'checkpoint.pt').write_text('saved on cancel')\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "os.kill(os.getppid(), signal.SIGTERM)\n"
            "time.sleep(30)\n"
            "(out / 'checkpoint.pt').write_text('never cancelled')\n"
        )})

        code = self.run_entrypoint(sys.executable, "long.py")

        self.assertEqual(code, 0)
        self.assertEqual(self.cloud.outputs(), {"episode-3/checkpoint.pt": b"saved on cancel"})

    def venv_python(self) -> str:
        """A fresh virtual environment's python, standing in for the image's,
        so pip installs somewhere it may (and offline: only local wheels)."""
        venv = self.work.parent / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        patcher = mock.patch.object(wildhat_run.sys, "executable", str(python))
        patcher.start()
        self.addCleanup(patcher.stop)
        return str(python)

    def test_requirements_are_installed_before_the_command_runs(self):
        python = self.venv_python()
        self.upload_source({
            "requirements.txt": "./wheels/greeting-1.0-py3-none-any.whl\n",
            "wheels/greeting-1.0-py3-none-any.whl": wheel("greeting", "1.0", "HELLO = 'hi from a requirement'\n"),
            "main.py": (
                "import os, pathlib, greeting\n"
                "pathlib.Path(os.environ['WILDHAT_OUTPUT_DIR'], 'said.txt').write_text(greeting.HELLO)\n"
            ),
        })

        code = self.run_entrypoint(python, "main.py")

        self.assertEqual(code, 0)
        self.assertEqual(self.cloud.outputs(), {"episode-3/said.txt": b"hi from a requirement"})

    def test_failing_requirements_stop_the_run(self):
        python = self.venv_python()
        self.upload_source({
            "requirements.txt": "./wheels/missing-1.0-py3-none-any.whl\n",
            "main.py": "raise SystemExit('ran without its requirements')",
        })

        code = self.run_entrypoint(python, "main.py")

        self.assertIn("pip install -r requirements.txt failed", str(code))

    def test_a_damaged_source_is_refused(self):
        self.upload_source({"main.py": "raise SystemExit('ran a damaged source')"}, damage=True)

        code = self.run_entrypoint(sys.executable, "main.py")

        self.assertIn("is damaged", str(code))
        self.assertFalse((self.work / "src").exists())

    def test_files_outside_the_source_folder_are_refused(self):
        self.upload_source({"../escaped.py": "", "main.py": ""})

        with self.assertRaises(tarfile.FilterError):
            self.run_entrypoint(sys.executable, "main.py")
        self.assertFalse((self.work / "escaped.py").exists())

    # --- Episodes ---------------------------------------------------------

    def report_env(self) -> dict:
        """Run a command that saves what the episode saw, and return it."""
        self.upload_source({"sim/report.py": (
            "import json, os, pathlib\n"
            "names = ['WILDHAT_EPISODE', 'WILDHAT_EPISODE_COUNT', 'WILDHAT_SEED', 'PYTHONUNBUFFERED']\n"
            "seen = {n: os.environ[n] for n in names}\n"
            "seen['cwd'] = pathlib.Path.cwd().name\n"
            "pathlib.Path(os.environ['WILDHAT_OUTPUT_DIR'], 'seen.json').write_text(json.dumps(seen))\n"
        )})
        self.env["WILDHAT_CWD"] = "sim"
        self.assertEqual(self.run_entrypoint(sys.executable, "report.py"), 0)
        return json.loads(self.cloud.outputs()["episode-3/seen.json"])

    def test_each_episode_gets_its_number_and_seed(self):
        seen = self.report_env()

        self.assertEqual(seen["WILDHAT_EPISODE"], "3")
        self.assertEqual(seen["WILDHAT_EPISODE_COUNT"], "5")
        self.assertEqual(seen["WILDHAT_SEED"], str(wildhat_run.episode_seed(BASE_SEED, 3)))
        # Otherwise print() output would only reach the logs when it exits.
        self.assertEqual(seen["PYTHONUNBUFFERED"], "1")
        # It runs where the person launched it from (ai/sim, say).
        self.assertEqual(seen["cwd"], "sim")

    def test_seeds_differ_by_episode_and_run(self):
        seeds = {wildhat_run.episode_seed(base, i) for base in (BASE_SEED, BASE_SEED + 1) for i in range(100)}
        self.assertEqual(len(seeds), 200)
        self.assertTrue(all(0 <= seed < 2**63 for seed in seeds))

    def test_a_replayed_episode_uses_the_given_seed(self):
        self.env["WILDHAT_SEED"] = "42"

        self.assertEqual(self.report_env()["WILDHAT_SEED"], "42")

    def test_a_folder_outside_the_source_is_refused(self):
        self.upload_source({"main.py": ""})
        self.env["WILDHAT_CWD"] = "../.."

        self.assertIn("there's no folder", str(self.run_entrypoint(sys.executable, "main.py")))

    # --- Run records --------------------------------------------------------

    def test_the_episode_record_follows_the_episode(self):
        self.upload_source({"main.py": ""})

        self.assertEqual(self.run_entrypoint(sys.executable, "main.py"), 0)

        first, last = self.cloud.updates[EPISODE_DOC]
        self.assertEqual(first["status"], "running")
        self.assertIn("started_at", first)
        self.assertEqual(last["status"], "succeeded")
        self.assertEqual(last["exit_code"], "0")
        self.assertEqual(last["outputs"], OUTPUT + "episode-3/")
        self.assertIn("finished_at", last)

    def test_failures_and_stops_are_recorded(self):
        self.upload_source({"crash.py": "raise SystemExit(7)"})
        self.run_entrypoint(sys.executable, "crash.py")
        self.assertEqual(self.cloud.episode()["status"], "failed")
        self.assertEqual(self.cloud.episode()["exit_code"], "7")

        if os.name != "nt":
            self.upload_source({"long.py": (
                "import os, signal, time\n"
                "signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
                "os.kill(os.getppid(), signal.SIGTERM)\n"
                "time.sleep(30)\n"
            )})
            shutil.rmtree(self.work)
            self.run_entrypoint(sys.executable, "long.py")
            self.assertEqual(self.cloud.episode()["status"], "stopped")

    def test_a_run_that_never_started_its_command_is_recorded_as_failed(self):
        self.upload_source({"main.py": ""}, damage=True)

        self.run_entrypoint(sys.executable, "main.py")

        self.assertEqual(self.cloud.episode()["status"], "failed")
        self.assertIn("is damaged", self.cloud.episode()["error"])

    def test_a_firestore_outage_doesnt_fail_the_episode(self):
        self.cloud.firestore_down = True
        self.upload_source({"main.py": (
            "import os, pathlib\n"
            "pathlib.Path(os.environ['WILDHAT_OUTPUT_DIR'], 'out.txt').write_text('done')\n"
        )})

        self.assertEqual(self.run_entrypoint(sys.executable, "main.py"), 0)
        self.assertEqual(self.cloud.outputs(), {"episode-3/out.txt": b"done"})

    def test_no_command(self):
        self.upload_source({"main.py": ""})

        self.assertIn("no command", str(self.run_entrypoint()))


if __name__ == "__main__":
    unittest.main()
