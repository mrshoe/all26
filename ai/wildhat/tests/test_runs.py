"""wildhat runs, logs, cancel and download, on runs started by the real
launch. The tests then play the cloud's part: the Cloud Run execution's
progress, the episode records the base image's entrypoint writes, its log
lines and its output files."""

import datetime
import io
import os
import tarfile
from unittest import mock

from harness import WILDHAT_BUCKET
from test_launch import LaunchTestCase

from wildhat import runs

DONE = "2026-10-26T08:00:00Z"


class RunsTestCase(LaunchTestCase):
    def setUp(self):
        super().setUp()
        os.environ["COLUMNS"] = "200"
        for name, value in [("FOLLOW_SECONDS", 0.01), ("LOG_DELAY_SECONDS", 1)]:
            patcher = mock.patch.object(runs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def start(self, *args: str, command=("python", "sim.py")) -> tuple[str, str]:
        """Launch a run; returns its ID and Cloud Run execution's name."""
        run = self.launch(*args, "--", *command)
        self.assertSucceeded(run)
        run_id = run.out.split("Launched run ")[1].split(":")[0]
        return run_id, self.google.documents[f"runs/{run_id}"]["execution"]

    def episode_did(self, run_id: str, index: int, status: str, exit_code: int | None = None,
                    started="2026-10-26T07:00:00Z", finished="2026-10-26T07:02:05.123456789Z") -> None:
        """What the base image's entrypoint records as an episode goes."""
        typed = {"status": {"stringValue": status}, "started_at": {"timestampValue": started}}
        if status not in ("pending", "running"):
            typed["finished_at"] = {"timestampValue": finished}
            typed["exit_code"] = {"integerValue": str(exit_code)}
        self.google.update_document(f"runs/{run_id}/episodes/{index}", typed)

    def upload_output(self, run_id: str, path: str, data: bytes) -> None:
        self.google.put_object(WILDHAT_BUCKET, f"runs/kai-morales/{run_id}/{path}", data)

    def someone_elses(self, run_id: str, handle="jane-doe") -> None:
        self.google.update_document(f"runs/{run_id}", {"launched_by": {"stringValue": handle}})

    def rows(self, out: str) -> dict[str, list[str]]:
        """`wildhat runs` output: each run's columns, by run ID."""
        lines = out.strip().splitlines()
        assert lines[0].split()[:3] == ["RUN", "BY", "STARTED"], out
        return {line.split()[0]: line for line in lines[1:]}


class RunsListTest(RunsTestCase):
    def test_lists_everyones_runs_newest_first_with_their_status(self):
        done, _ = self.start()
        self.google.set_execution(self.google.documents[f"runs/{done}"]["execution"], succeededCount=1, completionTime=DONE)
        running, exe = self.start("--episodes", "5")
        self.google.set_execution(exe, runningCount=3, succeededCount=1, failedCount=1)
        broken, exe = self.start("--episodes", "3")
        self.google.set_execution(exe, succeededCount=2, failedCount=1, completionTime=DONE)
        stopped, exe = self.start("--episodes", "2")
        self.google.set_execution(exe, succeededCount=1, cancelledCount=1, completionTime=DONE)
        waiting, _ = self.start(command=("python", "sim.py", "--fast"))
        self.someone_elses(waiting)

        run = self.wildhat("runs")

        self.assertSucceeded(run)
        rows = self.rows(run.out)
        self.assertEqual(list(rows), [waiting, stopped, broken, running, done][:len(rows)])
        self.assertIn("succeeded", rows[done])
        self.assertIn("running (2 of 5 done)", rows[running])
        self.assertIn("failed (1 of 3)", rows[broken])
        self.assertIn("cancelled", rows[stopped])
        self.assertIn("starting", rows[waiting])
        self.assertIn("jane-doe", rows[waiting])
        self.assertIn("python sim.py --fast", rows[waiting])

    def test_mine_and_one_persons(self):
        mine, _ = self.start()
        janes, _ = self.start()
        self.someone_elses(janes)

        self.assertEqual(list(self.rows(self.wildhat("runs", "--mine").out)), [mine])
        self.assertEqual(list(self.rows(self.wildhat("runs", "--user", "jane-doe").out)), [janes])
        self.assertIn("No runs by bob yet", self.wildhat("runs", "--user", "bob").out)
        self.assertNotEqual(self.wildhat("runs", "--mine", "--user", "bob").code, 0)

    def test_limit(self):
        ids = [self.start()[0] for _ in range(3)]

        self.assertEqual(len(self.rows(self.wildhat("runs", "--limit", "2").out)), 2)
        self.assertEqual(len(self.rows(self.wildhat("runs").out)), len(set(ids)))

    def test_no_runs(self):
        self.assertIn("No runs yet", self.wildhat("runs").out)

    def test_long_commands_fit_the_terminal(self):
        self.start(command=("python", "sim.py", *["--option"] * 40))
        os.environ["COLUMNS"] = "100"

        (line,) = self.rows(self.wildhat("runs").out).values()

        self.assertLessEqual(len(line), 100)
        self.assertTrue(line.endswith("…"))


class RunDetailTest(RunsTestCase):
    def test_shows_the_run_and_each_episode(self):
        run_id, exe = self.start("--episodes", "4", "--timeout", "90m")
        self.episode_did(run_id, 0, "succeeded", 0)
        self.episode_did(run_id, 1, "failed", 3)
        self.episode_did(run_id, 2, "stopped", 143)
        self.google.set_execution(exe, succeededCount=1, failedCount=2, cancelledCount=1, completionTime=DONE)

        run = self.wildhat("runs", run_id)

        self.assertSucceeded(run)
        record = self.google.documents[f"runs/{run_id}"]
        self.assertIn(f"Run {run_id}, launched by kai-morales", run.out)
        self.assertIn("status:   cancelled", run.out)
        self.assertIn("command:  python sim.py   (in ai/src/sim)", run.out)
        self.assertIn(f"code:     {record['git']['commit'][:7]} on main with uncommitted changes", run.out)
        # Where things are kept in the cloud isn't something people need.
        self.assertNotIn("gs://", run.out)
        self.assertIn("on:       CPU, up to 90 min per episode", run.out)
        episodes = {line.split()[0]: line.split()[1:] for line in run.out.split("EPISODE")[1].strip().splitlines()[1:]}
        seeds = [self.google.documents[f"runs/{run_id}/episodes/{i}"]["seed"] for i in range(4)]
        self.assertEqual(episodes["0"], [str(seeds[0]), "succeeded", "0", "2m05s"])
        self.assertEqual(episodes["1"], [str(seeds[1]), "failed", "3", "2m05s"])
        self.assertEqual(episodes["2"], [str(seeds[2]), "stopped", "143", "2m05s"])
        # Cancelled before it started, so it never updated its record.
        self.assertEqual(episodes["3"], [str(seeds[3]), "not", "started"])

    def test_an_episode_that_ended_without_reporting(self):
        run_id, exe = self.start()
        self.episode_did(run_id, 0, "running")
        self.google.set_execution(exe, failedCount=1, completionTime=DONE)

        self.assertIn("unknown", self.wildhat("runs", run_id).out.split("EPISODE")[1])

    def test_a_running_episode_is_running(self):
        run_id, exe = self.start()
        self.episode_did(run_id, 0, "running")
        self.google.set_execution(exe, runningCount=1)

        out = self.wildhat("runs", run_id).out
        self.assertIn("status:   running", out)
        self.assertIn("running", out.split("EPISODE")[1])

    def test_bad_or_missing_run_ids(self):
        self.assertFailedWith(self.wildhat("runs", "nhrj"), "isn't a run ID")
        self.assertFailedWith(self.wildhat("runs", "20261026-0000-zzzz"), "There's no run 20261026-0000-zzzz")


class LogsTest(RunsTestCase):
    def test_prints_every_episodes_lines_with_its_number(self):
        run_id, exe = self.start("--episodes", "3")
        self.google.add_log(exe, None, None, log="cloudaudit.googleapis.com%2Fsystem_event")
        for i, (episode, text) in enumerate([(0, "zero a"), (2, "two a"), (0, "zero b"), (1, "one a"), (2, "two b")]):
            self.google.add_log(exe, episode, text + "\n")
        self.google.add_log(exe, 1, "Container called exit(0).", log="run.googleapis.com%2Fvarlog%2Fsystem")
        self.google.add_log(self.start()[1], 0, "another run's line")

        run = self.wildhat("logs", run_id)

        self.assertSucceeded(run)
        # Over several pages of two; not the audit log's entry.
        self.assertEqual(run.out.splitlines(), [
            "[0] zero a", "[2] two a", "[0] zero b", "[1] one a", "[2] two b", "[1] Container called exit(0).",
        ])

    def test_one_episode(self):
        run_id, exe = self.start("--episodes", "3")
        self.google.add_log(exe, 0, "zero")
        self.google.add_log(exe, 2, "two")

        self.assertEqual(self.wildhat("logs", run_id, "--episode", "2").out.splitlines(), ["two"])
        self.assertFailedWith(self.wildhat("logs", run_id, "--episode", "3"), "has episodes 0 to 2")

    def test_a_single_episode_run_has_no_prefixes(self):
        run_id, exe = self.start()
        self.google.add_log(exe, 0, "hello")

        self.assertEqual(self.wildhat("logs", run_id).out.splitlines(), ["hello"])

    def test_follow_prints_new_lines_until_the_run_ends(self):
        run_id, exe = self.start("--episodes", "2")
        self.google.add_log(exe, 0, "first")
        just_now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        script = iter([
            lambda: self.google.add_log(exe, 1, "second"),
            lambda: None,
            lambda: self.google.add_log(exe, 0, "third"),
            lambda: self.google.set_execution(exe, succeededCount=2, completionTime=just_now),
            # Cloud Logging takes a while to make the last lines searchable.
            lambda: None,
            lambda: self.google.add_log(exe, 1, "last words"),
        ])
        # Each time logs are read, the run moves along (pages ask again, too).
        self.google.on_log_read = lambda: next(script, lambda: None)()

        run = self.wildhat("logs", run_id, "-f")

        self.assertSucceeded(run)
        lines = run.out.splitlines()
        for text in ("[0] first", "[1] second", "[0] third", "[1] last words"):
            self.assertEqual(lines.count(text), 1, run.out)

    def test_follow_catches_lines_that_show_up_out_of_order(self):
        # Seen on a real run: Cloud Logging made an episode's later lines
        # searchable before its earlier ones.
        run_id, exe = self.start("--episodes", "2")
        self.google.add_log(exe, 0, "zero, second 1", second=1)
        self.google.add_log(exe, 1, "one, second 9", second=9)
        just_now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        script = iter([
            lambda: None,
            lambda: self.google.add_log(exe, 1, "one, second 4", second=4),  # Late.
            lambda: self.google.set_execution(exe, succeededCount=2, completionTime=just_now),
        ])
        self.google.on_log_read = lambda: next(script, lambda: None)()

        run = self.wildhat("logs", run_id, "-f")

        self.assertSucceeded(run)
        self.assertEqual(sorted(run.out.splitlines()), ["[0] zero, second 1", "[1] one, second 4", "[1] one, second 9"])

    def test_following_a_run_that_ended_long_ago_doesnt_wait(self):
        run_id, exe = self.start()
        self.google.add_log(exe, 0, "done")
        self.google.set_execution(exe, succeededCount=1, completionTime=DONE)

        run = self.wildhat("logs", run_id, "-f")

        self.assertEqual(run.out.splitlines(), ["done"])
        self.assertEqual(len(self.google.requests_to("entries:list")), 1)

    def test_without_follow_it_doesnt_wait(self):
        run_id, exe = self.start()
        self.google.set_execution(exe, runningCount=1)

        self.assertIn("No logs yet", self.wildhat("logs", run_id).out)

    def test_a_run_that_never_started(self):
        run_id, _ = self.start()
        self.google.update_document(f"runs/{run_id}", {"execution": {"nullValue": None}})

        self.assertFailedWith(self.wildhat("logs", run_id), "never started")


class CancelTest(RunsTestCase):
    def test_asks_then_cancels(self):
        run_id, exe = self.start("--episodes", "3")
        self.google.set_execution(exe, runningCount=2, succeededCount=1)

        run = self.wildhat("cancel", run_id, typed=["y"])

        self.assertSucceeded(run)
        self.assertIn(f"Cancel run {run_id}: python sim.py [running (1 of 3 done)]? [y/N]", run.out)
        self.assertEqual(self.google.cancelled_executions, [exe])
        self.assertIn("cancelled", self.rows(self.wildhat("runs").out)[run_id])

    def test_saying_no_cancels_nothing(self):
        run_id, exe = self.start()
        self.google.set_execution(exe, runningCount=1)

        for typed in (["n"], []):
            run = self.wildhat("cancel", run_id, typed=typed)
            self.assertNotEqual(run.code, 0)
        self.assertEqual(self.google.cancelled_executions, [])

    def test_yes_skips_the_question(self):
        run_id, exe = self.start()

        run = self.wildhat("cancel", run_id, "--yes")

        self.assertSucceeded(run)
        self.assertNotIn("[y/N]", run.out)
        self.assertEqual(self.google.cancelled_executions, [exe])

    def test_only_your_own(self):
        run_id, _ = self.start()
        self.someone_elses(run_id)

        self.assertFailedWith(self.wildhat("cancel", run_id, "--yes"), "you can only cancel your own")
        self.assertEqual(self.google.cancelled_executions, [])

    def test_cloud_run_also_refuses_someone_elses_job(self):
        # Even if the record is edited to say it's yours.
        run_id, exe = self.start()
        self.google.update_document(f"runs/{run_id}", {"job": {"stringValue": "wh-jane-doe"}})
        state = self.gcloud_state()
        state["executions"][0]["metadata"]["name"] = exe.replace("wh-kai-morales", "wh-jane-doe")
        (self.fake_dir / "gcloud.json").write_text(__import__("json").dumps(state), encoding="utf-8")
        self.google.update_document(f"runs/{run_id}", {"execution": {"stringValue": exe.replace("wh-kai-morales", "wh-jane-doe")}})

        self.assertFailedWith(self.wildhat("cancel", run_id, "--yes"), "HTTP 403")
        self.assertEqual(self.google.cancelled_executions, [])

    def test_an_ended_run(self):
        run_id, exe = self.start()
        self.google.set_execution(exe, succeededCount=1, completionTime=DONE)

        run = self.wildhat("cancel", run_id, "--yes")

        self.assertSucceeded(run)
        self.assertIn("has already ended (succeeded)", run.out)
        self.assertEqual(self.google.cancelled_executions, [])


class DownloadTest(RunsTestCase):
    def setUp(self):
        super().setUp()
        self.run_id, _ = self.start("--episodes", "2")
        self.upload_output(self.run_id, "episode-0/weights/best.pt", b"zero")
        self.upload_output(self.run_id, "episode-0/results.csv", b"a,b\n")
        self.upload_output(self.run_id, "episode-1/results.csv", b"c,d\n")
        self.away = self.tmp / "downloads here"
        self.away.mkdir()
        self.cd(self.away)

    def files(self, folder) -> dict[str, bytes]:
        return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def test_downloads_every_episodes_outputs(self):
        run = self.wildhat("download", self.run_id)

        self.assertSucceeded(run)
        self.assertEqual(self.files(self.away / self.run_id), {
            "episode-0/weights/best.pt": b"zero",
            "episode-0/results.csv": b"a,b\n",
            "episode-1/results.csv": b"c,d\n",
        })
        self.assertIn(f"Downloaded 3 file(s) (12 B) to {self.run_id}", run.out)

    def test_one_episode_somewhere_else(self):
        self.assertSucceeded(self.wildhat("download", self.run_id, "--episode", "1", "--to", "ep1"))

        self.assertEqual(self.files(self.away / "ep1"), {"episode-1/results.csv": b"c,d\n"})

    def test_the_code_it_ran(self):
        run = self.wildhat("download", self.run_id, "--source")

        self.assertSucceeded(run)
        name = self.google.documents[f"runs/{self.run_id}"]["source"].removeprefix(f"gs://{WILDHAT_BUCKET}/")
        with tarfile.open(fileobj=io.BytesIO(self.google.object(WILDHAT_BUCKET, name))) as tar:
            packed = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        self.assertEqual(self.files(self.away / self.run_id / "source"), packed)
        self.assertIn(b"print('sim, edited')\n", packed.values())

    def test_not_into_ai_src(self):
        self.cd(self.src / "sim")

        self.assertFailedWith(self.wildhat("download", self.run_id), "Not downloading into ai/src")
        self.assertFalse((self.src / "sim" / self.run_id).exists())
        # Somewhere else is fine, even from there.
        self.assertSucceeded(self.wildhat("download", self.run_id, "--to", str(self.away / "x")))

    def test_no_outputs(self):
        self.cd(self.src / "sim")
        quiet, _ = self.start()
        self.cd(self.away)

        self.assertIn(f"Run {quiet} has no outputs.", self.wildhat("download", quiet).out)
        self.assertIn("has no outputs for episode 0.", self.wildhat("download", quiet, "--episode", "0").out)


class TimesTest(RunsTestCase):
    def test_reads_firestore_and_cloud_run_times(self):
        for stamp, micro in [("2026-10-26T06:46:28.664906Z", 664906), ("2026-10-26T06:46:28Z", 0),
                             ("2026-10-26T07:24:11.691053123Z", 691053), ("2026-10-26T06:46:28.5-07:00", 500000)]:
            self.assertEqual(runs.parse_time(stamp).microsecond, micro, stamp)
