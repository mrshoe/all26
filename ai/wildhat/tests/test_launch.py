"""wildhat launch: pack the ai folder, record the run in Firestore, start the
person's Cloud Run job. Uses a real git repository in a temp folder."""

import hashlib
import importlib.util
import io
import os
import re
import subprocess
import tarfile
from pathlib import Path
from unittest import mock

from harness import PROJECT, WILDHAT_BUCKET, WildhatTestCase

from wildhat import launch

SA = f"wh-kai-morales@{PROJECT}.iam.gserviceaccount.com"
RUN_ID = re.compile(r"^\d{8}-\d{4}-[a-km-z2-9]{4}$")

ENTRYPOINT = Path(__file__).resolve().parents[2] / "images" / "base" / "wildhat_run.py"


class LaunchTest(WildhatTestCase):
    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "all26 repo"
        self.src = self.repo / "ai" / "src"
        self.write({
            "ai/src/.gitignore": "*.log\n",
            "ai/src/requirements.txt": "",
            "ai/src/sim/sim.py": "print('sim')\n",
            "ai/src/shared/util.py": "X = 1\n",
            "ai/src/shared/old.py": "gone\n",
            # Next to ai/src, and never uploaded.
            "ai/README.md": "docs\n",
            "ai/wildhat/cli.py": "tooling\n",
            "ai/terraform/main.tf": "tooling\n",
            "lib/Robot.java": "not ai\n",
        })
        self.git("init", "-q", "-b", "main")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "start")
        self.commit = self.git("rev-parse", "HEAD").strip()
        # What a working tree usually looks like: edits, a new file, a
        # deleted one, and ignored junk.
        self.write({
            "ai/src/sim/sim.py": "print('sim, edited')\n",
            "ai/src/sim/new.py": "new\n",
            "ai/src/sim/run.log": "junk\n",
            "ai/private-notes.md": "a student's email\n",
        })
        (self.src / "shared" / "old.py").unlink()

        self.cd(self.src / "sim")
        self.log_in()

    def write(self, files: dict[str, str]) -> None:
        for name, text in files.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # \n everywhere: on Windows, text mode would write \r\n.
            path.write_text(text, encoding="utf-8", newline="\n")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "core.autocrlf=false", *args],
            cwd=self.repo, capture_output=True, text=True, check=True,
        ).stdout

    def cd(self, path: Path) -> None:
        old = os.getcwd()
        os.chdir(path)
        self.addCleanup(os.chdir, old)

    def launch(self, *args: str):
        return self.wildhat("launch", *args)

    def executions(self) -> list[dict]:
        return self.gcloud_state().get("executions", [])

    def env_of(self, execution: dict) -> dict:
        return {e["name"]: e["value"] for e in execution["spec"]["template"]["spec"]["containers"][0]["env"]}

    def run_record(self) -> tuple[str, dict, dict[int, dict]]:
        runs = {k.split("/")[1] for k in self.google.documents if k.startswith("runs/")}
        (run_id,) = runs
        run = self.google.documents[f"runs/{run_id}"]
        episodes = {
            int(k.rsplit("/", 1)[1]): v
            for k, v in self.google.documents.items()
            if k.startswith(f"runs/{run_id}/episodes/")
        }
        return run_id, run, episodes

    def tarball(self) -> tuple[str, dict[str, bytes]]:
        (name,) = [n for (b, n) in self.google.objects if b == WILDHAT_BUCKET and n.startswith("src/")]
        data = self.google.object(WILDHAT_BUCKET, name)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        return name, files

    # --- What gets packed ----------------------------------------------------

    def test_packs_ai_src_as_it_is_on_disk(self):
        run = self.launch("--", "python", "sim.py")

        self.assertSucceeded(run)
        name, files = self.tarball()
        self.assertEqual(files, {
            ".gitignore": b"*.log\n",
            "requirements.txt": b"",
            "sim/sim.py": b"print('sim, edited')\n",
            "sim/new.py": b"new\n",
            "shared/util.py": b"X = 1\n",
        })
        data = self.google.object(WILDHAT_BUCKET, name)
        self.assertEqual(name, f"src/kai-morales/{hashlib.sha256(data).hexdigest()}.tar.gz")
        self.assertIn("Packed 5 files", run.out)
        self.assertIn("from ai/src/", run.out)
        self.assertIn(f"at {self.commit[:7]}, with uncommitted changes", run.out)
        # Everyone can read it, so new files are named in case one is private.
        self.assertIn("Included 1 file(s) git doesn't track yet: sim/new.py", run.out)

    def test_the_same_files_pack_the_same_and_upload_once(self):
        self.assertSucceeded(self.launch("--", "python", "sim.py"))
        os.utime(self.src / "sim" / "sim.py", (0, 0))  # Timestamps don't count.

        run = self.launch("--", "python", "sim.py")

        self.assertSucceeded(run)
        self.assertIn("already uploaded", run.out)
        uploads = [r for r in self.google.requests if r["path"].startswith("/upload/")]
        self.assertEqual(len(uploads), 1)

    def test_too_big_to_pack(self):
        with mock.patch.object(launch, "MAX_SOURCE_BYTES", 100):
            run = self.launch("--", "python", "sim.py")

        self.assertFailedWith(run, "over the 100 B limit")
        self.assertEqual(self.google.requests_to(":commit"), [])

    def test_only_from_inside_ai_src(self):
        for folder in (self.repo / "ai", self.repo / "lib", self.tmp):  # tmp isn't in git at all
            with self.subTest(folder=folder.name):
                self.cd(folder)
                self.assertFailedWith(self.launch("--", "python", "sim.py"), "from inside ai/src/")
        self.assertEqual(self.google.requests_to(":commit"), [])

    # --- The run record ----------------------------------------------------

    def test_records_the_run_and_its_episodes(self):
        run = self.launch("--episodes", "3", "--timeout", "90m", "--", "python", "sim.py", "--speed", "1,5")

        self.assertSucceeded(run)
        run_id, record, episodes = self.run_record()
        self.assertRegex(run_id, RUN_ID)
        self.assertIn(f"Launched run {run_id}: 3 episodes on CPU, up to 90 min each.", run.out)
        name, _ = self.tarball()
        self.assertEqual(record["source"], f"gs://{WILDHAT_BUCKET}/{name}")
        self.assertEqual(record["source_sha256"], name.split("/")[-1].removesuffix(".tar.gz"))
        self.assertEqual(record["git"], {"commit": self.commit, "branch": "main", "dirty": True})
        (execution,) = self.executions()
        for field, value in {
            "launched_by": "kai-morales",
            "command": ["python", "sim.py", "--speed", "1,5"],
            "cwd": "sim",
            "episodes": 3,
            "gpu": False,
            "timeout_seconds": 5400,
            "job": "wh-kai-morales",
            "execution": execution["metadata"]["name"],
            "outputs": f"gs://{WILDHAT_BUCKET}/runs/kai-morales/{run_id}/",
        }.items():
            self.assertEqual(record[field], value, field)
        self.assertEqual(sorted(episodes), [0, 1, 2])
        for i, episode in episodes.items():
            self.assertEqual(episode["status"], "pending")
            self.assertEqual(episode["seed"], launch.episode_seed(record["base_seed"], i))
        self.assertEqual(len({e["seed"] for e in episodes.values()}), 3)

    def test_a_clean_tree_isnt_dirty(self):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "everything")

        run = self.launch("--", "python", "sim.py")

        self.assertSucceeded(run)
        self.assertFalse(self.run_record()[1]["git"]["dirty"])
        self.assertNotIn("uncommitted", run.out)

    def test_the_images_entrypoint_derives_the_same_seeds(self):
        spec = importlib.util.spec_from_file_location("wildhat_run", ENTRYPOINT)
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)

        for base in (0, 1, 2**63 - 1, 7364523456):
            for i in (0, 1, 99):
                self.assertEqual(entrypoint.episode_seed(base, i), launch.episode_seed(base, i))

    # --- Starting the job --------------------------------------------------

    def test_starts_the_persons_job_with_one_task_per_episode(self):
        self.assertSucceeded(self.launch("--episodes", "4", "--", "python", "sim.py", "--speed", "1,5", "a|b"))

        run_id, record, _ = self.run_record()
        (execute,) = [c for c in self.calls("gcloud") if c["args"][:3] == ["run", "jobs", "execute"]]
        self.assertEqual(execute["args"][3], "wh-kai-morales")
        self.assertIn(f"--account={SA}", execute["args"])
        self.assertIn("--async", execute["args"])
        (execution,) = self.executions()
        spec = execution["spec"]
        self.assertEqual(spec["taskCount"], 4)
        self.assertEqual(spec["template"]["spec"]["timeoutSeconds"], "3600")
        # Commas and | in arguments survive gcloud's list syntax.
        self.assertEqual(spec["template"]["spec"]["containers"][0]["args"], ["python", "sim.py", "--speed", "1,5", "a|b"])
        self.assertEqual(self.env_of(execution), {
            "WILDHAT_SOURCE": record["source"],
            "WILDHAT_OUTPUT": record["outputs"],
            "WILDHAT_RUN_DOC": f"projects/{PROJECT}/databases/(default)/documents/runs/{run_id}",
            "WILDHAT_CWD": "sim",
            "WILDHAT_EPISODE_COUNT": "4",
            "WILDHAT_BASE_SEED": str(record["base_seed"]),
        })

    def test_from_the_top_of_ai_src(self):
        self.cd(self.src)

        self.assertSucceeded(self.launch("--", "python", "sim/sim.py"))

        self.assertEqual(self.run_record()[1]["cwd"], "")

    def test_replaying_an_episode_with_its_seed(self):
        self.assertSucceeded(self.launch("--seed", "42", "--", "python", "sim.py"))

        _, _, episodes = self.run_record()
        self.assertEqual(episodes[0]["seed"], 42)
        self.assertEqual(self.env_of(self.executions()[0])["WILDHAT_SEED"], "42")

    def test_gpu_runs_use_the_gpu_job(self):
        state = self.gcloud_state()
        state["gpu_jobs"] = True
        (self.fake_dir / "gcloud.json").write_text(__import__("json").dumps(state), encoding="utf-8")

        run = self.launch("--gpu", "--", "python", "train.py")

        self.assertSucceeded(run)
        self.assertIn("1 episode on a GPU", run.out)
        self.assertEqual(self.run_record()[1]["job"], "wh-kai-morales-gpu")

    def test_gpu_runs_before_there_are_gpu_jobs(self):
        run = self.launch("--gpu", "--", "python", "train.py")

        self.assertFailedWith(run, "GPU jobs aren't set up yet")
        self.assertEqual(self.google.documents, {})  # The records are cleaned up.

    def test_a_failure_to_start_cleans_up_the_records(self):
        self.respond("gcloud", ["run", "jobs", "execute"], stderr="ERROR: RESOURCE_EXHAUSTED: quota", returncode=1)

        run = self.launch("--episodes", "2", "--", "python", "sim.py")

        self.assertFailedWith(run, "Couldn't start the run")
        self.assertIn("RESOURCE_EXHAUSTED", run.err)
        self.assertEqual(self.google.documents, {})

    # --- Limits, and mistakes ------------------------------------------------

    def test_limits_are_checked_before_anything_happens(self):
        for args, message in [
            (["--episodes", "101"], "between 1 and 100"),
            (["--episodes", "0"], "between 1 and 100"),
            (["--timeout", "9h"], "at most 8h"),
            (["--timeout", "soon"], "Can't read the timeout"),
            (["--seed", "1", "--episodes", "2"], "needs --episodes 1"),
            (["--seed", "-1"], "between 0 and 2^63 - 1"),
        ]:
            with self.subTest(args=args):
                before = len(self.google.requests)
                self.assertFailedWith(self.launch(*args, "--", "python", "sim.py"), message)
                self.assertEqual(len(self.google.requests), before)

    def test_timeouts(self):
        for text, seconds in [("90m", 5400), ("2h", 7200), ("45s", 45), ("600", 600), ("8h", 28800)]:
            with self.subTest(text=text):
                self.assertEqual(launch.parse_duration(text), seconds)

    def test_a_command_is_needed(self):
        self.assertFailedWith(self.launch(), "Say what to run")
        self.assertFailedWith(self.launch("--"), "Say what to run")

    def test_the_command_can_skip_the_double_dash(self):
        self.assertSucceeded(self.launch("python", "sim.py"))

        self.assertEqual(self.run_record()[1]["command"], ["python", "sim.py"])

    def test_not_signed_in(self):
        self.assertSucceeded(self.wildhat("logout"))

        self.assertFailedWith(self.launch("--", "python", "sim.py"), "You're not signed in")
