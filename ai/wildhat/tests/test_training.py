"""wildhat training list, show, tag, download and sync, on records and files
like the ones ai/src/shared/training.py writes."""

import datetime
import json
import os
import time
from unittest import mock

from harness import WILDHAT_BUCKET, WildhatTestCase

from wildhat.launch import firestore_value

NOW = datetime.datetime.now(datetime.timezone.utc)
A, B, C = "20261101-0900-aaaa", "20261102-0900-bbbb", "20261103-0900-cccc"


def ago(minutes: float) -> datetime.datetime:
    return NOW - datetime.timedelta(minutes=minutes)


class TrainingTestCase(WildhatTestCase):
    def setUp(self):
        super().setUp()
        self.log_in()
        patcher = mock.patch.dict(os.environ, {"WILDHAT_CACHE": str(self.tmp / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def put(self, path: str, fields: dict) -> None:
        self.google.update_document(path, firestore_value(fields)["mapValue"]["fields"])

    def training_run(self, run_id: str, model="fuel_detector", who="kai-morales", status="finished",
                     created=None, **more) -> None:
        self.put(f"training/{run_id}", {
            "model": model, "trained_by": who, "status": status, "created_at": created or ago(60),
            "heartbeat_at": ago(1), "finished_at": None if status == "running" else ago(30), "error": None,
            "where": {"kind": "laptop", "host": "kai-mac"},
            "segments": [{"source": "gs://b/src/x.tar.gz", "source_sha256": "abc", "started_at": ago(60),
                          "git": {"commit": "52bd9dd3fb04", "branch": "main", "dirty": True},
                          "where": {"kind": "laptop", "host": "kai-mac"}}],
            "dataset": {"path": "training/fuel_detector/dataset.toml", "samples": {"train": 900, "validation": 100}},
            "parents": {}, "parent_runs": [], "step": 300, "latest": {"loss": 0.25},
            "best": {"step": 200, "metrics": {"val_ap50": 0.7}}, "last_checkpoint": 300, "checkpoints": 2,
            **more})

    def checkpoint(self, run_id: str, step: int, data=b"weights", who="kai-morales", **more) -> None:
        name = f"training/{who}/{run_id}/checkpoints/{step}/model.pt"
        self.google.put_object(WILDHAT_BUCKET, name, data)
        self.put(f"training/{run_id}/checkpoints/{step}", {
            "step": step, "file": name, "size": len(data), "md5": "x", "resumable": False, "metrics": {},
            "best": False, "saved_at": ago(40), "tags": [], "pruned": False, **more})

    def rows(self, out: str) -> dict[str, list[str]]:
        lines = out.strip().splitlines()
        return {line.split()[0]: line.split() for line in lines[1:]}


class ListTest(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.training_run(A, created=ago(300))
        self.training_run(B, model="reef_keypoints", who="dana-wells", created=ago(200))
        self.training_run(C, status="running", created=ago(100), heartbeat_at=ago(10))

    def test_newest_first(self):
        run = self.wildhat("training", "list")
        self.assertSucceeded(run)
        self.assertEqual(list(self.rows(run.out)), [C, B, A])
        self.assertEqual(self.rows(run.out)[A][1:3], ["fuel_detector", "kai-morales"])
        self.assertIn("loss 0.25", run.out)

    def test_silent_running_runs_have_probably_stopped(self):
        self.assertIn("stopped?", self.rows(self.wildhat("training", "list").out)[C])

    def test_filters(self):
        self.assertEqual(list(self.rows(self.wildhat("training", "list", "--model", "fuel_detector").out)), [C, A])
        self.assertEqual(list(self.rows(self.wildhat("training", "list", "--user", "dana-wells").out)), [B])
        self.assertEqual(list(self.rows(self.wildhat("training", "list", "--mine").out)), [C, A])
        self.assertEqual(list(self.rows(self.wildhat("training", "list", "--model", "reef_keypoints",
                                                      "--mine").out)), [])
        self.assertIn("No training runs of reef_keypoints by kai-morales yet.",
                      self.wildhat("training", "list", "--model", "reef_keypoints", "--mine").out)


class ShowTest(TrainingTestCase):
    def test_everything_about_a_run(self):
        self.training_run(A)
        self.checkpoint(A, 200, best=True, resumable=True, metrics={"val_ap50": 0.7}, tags=["fuel_detector:v1"])
        self.checkpoint(A, 100, resumable=True, pruned=True, best=True)  # best when saved, not now
        self.training_run(B, model="robot_vision", parents={"backbone": {
            "ref": "fuel_detector:v1", "checkpoint": f"{A}/200", "model": "fuel_detector", "file": "f", "md5": "x"}},
            parent_runs=[A])

        run = self.wildhat("training", "show", A)
        self.assertSucceeded(run)
        self.assertIn(f"Training run {A}: fuel_detector, by kai-morales", run.out)
        self.assertIn("code:     52bd9dd on main with uncommitted changes, laptop kai-mac", run.out)
        self.assertIn("dataset:  training/fuel_detector/dataset.toml (samples: train 900, validation 100)", run.out)
        self.assertIn(f"used by:  {B} (robot_vision)", run.out)
        self.assertIn("best:     step 200   val_ap50 0.7", run.out)
        self.assertRegex(run.out, rf"{A}/100 .* resumable, pruned")
        self.assertRegex(run.out, rf"{A}/200 .* best, resumable +val_ap50 0.7 +fuel_detector:v1")

        child = self.wildhat("training", "show", B)
        self.assertIn(f"started from: backbone = {A}/200, fuel_detector (asked as fuel_detector:v1)", child.out)

    def test_resumed_runs_show_each_segment(self):
        self.training_run(A)
        record = self.google.documents[f"training/{A}"]
        segments = record["segments"] + [{**record["segments"][0], "resumed_at_step": 4000,
                                          "where": {"kind": "cloud", "run": "20261101-1000-zzzz", "episode": 0}}]
        self.put(f"training/{A}", {"segments": segments})
        out = self.wildhat("training", "show", A).out
        self.assertIn("resumed at step 4000: 52bd9dd on main with uncommitted changes, cloud run 20261101-1000-zzzz",
                      out)

    def test_final_results_and_kept_files(self):
        self.training_run(A, final={"test_ap50": 0.91},
                          files=[{"name": "results.png", "object": "x", "size": 3, "content_type": "image/png"}])
        out = self.wildhat("training", "show", A).out
        self.assertIn("final:    test_ap50 0.91", out)
        self.assertIn("files:    results.png", out)

    def test_no_such_run(self):
        self.assertFailedWith(self.wildhat("training", "show", C), f"There's no training run {C}.")
        self.assertFailedWith(self.wildhat("training", "show", "latest"), "isn't a training run")


class TagTest(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.training_run(A)
        self.checkpoint(A, 200)
        self.checkpoint(A, 300)

    def tags(self) -> dict:
        return self.google.documents.get("models/fuel_detector", {}).get("tags", {})

    def test_tagging_a_run_means_its_best(self):
        run = self.wildhat("training", "tag", A, "fuel_detector:v1")
        self.assertSucceeded(run)
        self.assertIn(f"fuel_detector:v1 is now {A}/200.", run.out)
        self.assertEqual(self.tags(), {"v1": f"{A}/200"})
        self.assertEqual(self.google.documents[f"training/{A}/checkpoints/200"]["tags"], ["fuel_detector:v1"])

    def test_moving_a_tag(self):
        self.wildhat("training", "tag", A, "fuel_detector:v1")
        self.wildhat("training", "tag", A, "fuel_detector:v1.5")  # dots are fine
        run = self.wildhat("training", "tag", f"{A}/300", "fuel_detector:v1")
        self.assertIn(f"fuel_detector:v1 is now {A}/300 (it was {A}/200).", run.out)
        self.assertEqual(self.tags(), {"v1": f"{A}/300", "v1.5": f"{A}/200"})
        self.assertEqual(self.google.documents[f"training/{A}/checkpoints/200"]["tags"], ["fuel_detector:v1.5"])
        self.assertEqual(self.google.documents[f"training/{A}/checkpoints/300"]["tags"], ["fuel_detector:v1"])
        self.assertIn("is " + f"{A}/300 already", self.wildhat("training", "tag", f"{A}/300", "fuel_detector:v1").out)

    def test_mistakes(self):
        self.assertFailedWith(self.wildhat("training", "tag", A, "reef_keypoints:v1"),
                              "is a fuel_detector checkpoint, so it can't be reef_keypoints:v1")
        self.assertFailedWith(self.wildhat("training", "tag", A, "v1"), "isn't a tag")
        self.assertFailedWith(self.wildhat("training", "tag", f"{A}/250", "fuel_detector:v1"),
                              "has no checkpoint at step 250")
        self.put(f"training/{A}/checkpoints/300", {"pruned": True})
        self.assertFailedWith(self.wildhat("training", "tag", f"{A}/300", "fuel_detector:v1"),
                              "was deleted to save space")


class DownloadTest(TrainingTestCase):
    def test_a_tagged_checkpoint(self):
        self.training_run(A)
        self.checkpoint(A, 200, data=b"the weights")
        self.wildhat("training", "tag", A, "fuel_detector:v1")
        self.cd_to(self.tmp / "away")

        run = self.wildhat("training", "download", "fuel_detector:v1")
        self.assertSucceeded(run)
        self.assertEqual((self.tmp / "away" / "model.pt").read_bytes(), b"the weights")
        self.assertIn(f"Downloaded {A}/200", run.out)

    def cd_to(self, folder):
        folder.mkdir()
        old = os.getcwd()
        os.chdir(folder)
        self.addCleanup(os.chdir, old)


class SyncTest(TrainingTestCase):
    """Queues as shared/training.py leaves them when it can't send."""

    def queue(self, run_id: str, lock_age: float | None = None):
        folder = self.tmp / "cache" / "training" / run_id / "queue"
        folder.mkdir(parents=True)
        (folder / "000000-model.pt").write_bytes(b"queued weights")
        name = f"training/kai-morales/{run_id}/checkpoints/10/model.pt"
        (folder / "000000.json").write_text(json.dumps({
            "uploads": [{"object": name, "file": "000000-model.pt"}], "checkpoint": True,
            "patches": [{"doc": f"training/{run_id}", "mask": ["last_checkpoint"],
                         "fields": {"last_checkpoint": {"integerValue": "10"}}}]}))
        (folder / "000001.json").write_text(json.dumps({
            "uploads": [{"object": f"training/kai-morales/{run_id}/metrics/000003.jsonl", "text": "{}\n",
                         "content_type": "application/x-ndjson"}],
            "patches": [{"doc": f"training/{run_id}", "mask": ["status"],
                         "fields": {"status": {"stringValue": "finished"}}}]}))
        if lock_age is not None:
            (folder / ".lock").touch()
            stamp = time.time() - lock_age  # now, not when the tests started
            os.utime(folder / ".lock", (stamp, stamp))
        return folder

    def test_sends_whats_waiting_in_order(self):
        self.training_run(A, status="running")
        folder = self.queue(A, lock_age=600)  # a script that died long ago

        run = self.wildhat("training", "sync")
        self.assertSucceeded(run)
        self.assertIn(f"{A}: sent 2 waiting upload(s).", run.out)
        self.assertEqual(self.google.object(WILDHAT_BUCKET, f"training/kai-morales/{A}/checkpoints/10/model.pt"),
                         b"queued weights")
        record = self.google.documents[f"training/{A}"]
        self.assertEqual((record["status"], record["last_checkpoint"]), ("finished", 10))
        self.assertFalse(folder.exists())
        self.assertIn("Nothing is waiting to be sent.", self.wildhat("training", "sync").out)

    def test_leaves_a_running_scripts_queue_alone(self):
        self.training_run(A, status="running")
        self.queue(A, lock_age=5)
        run = self.wildhat("training", "sync")
        self.assertIn("still sending (2 waiting)", run.out)
        self.assertEqual(self.google.documents[f"training/{A}"]["status"], "running")


class HelpTest(TrainingTestCase):
    def test_listed_under_training(self):
        out = self.wildhat("--help").out
        self.assertIn("Training:", out)
        self.assertIn("training", out.split("Training:")[1])
