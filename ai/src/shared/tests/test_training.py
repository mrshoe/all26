"""Training runs: the record, metrics, checkpoints, [start] references,
background uploads that wait out going offline, pruning, resuming, and how
a run ends. The cloud is faked (from test_datasets), and so is packing the
source, which `wildhat pack-source` does. Run from ai/src:

    python -m unittest discover -s shared/tests -t .
"""

import atexit
import datetime
import json
import shutil
import signal
import sys
import threading
from pathlib import Path
from unittest import mock

from shared import training
from shared.cloud import plain
from shared.tests.test_datasets import DatasetTestCase, FakeCloud
from shared.training import TrainingRun, jobs_in, queue_folder, send_job

RUN = "20261102-0019-w22f"
SOURCE = {"source": "gs://test-project-wildhat/src/kai-morales/abc.tar.gz", "source_sha256": "abc",
          "git": {"commit": "e86e8f6", "branch": "main", "dirty": True}}


class FakeTrainingCloud(FakeCloud):
    """FakeCloud, plus what training writes. Thread-safe enough: the
    background sender and the test both use it."""

    handle = "kai-morales"

    def __init__(self, root: Path):
        super().__init__(root)
        self.uploads: list[str] = []
        self.deleted: list[str] = []
        self.offline = False
        self.lock = threading.Lock()

    def check_online(self):
        if self.offline:
            raise training.CloudError("Couldn't reach Google")

    def create(self, path, fields):
        with self.lock:
            assert path not in self.docs, f"{path} exists"
            self.docs[path] = json.loads(json.dumps(fields, default=str))

    def update(self, path, fields):
        self.check_online()
        with self.lock:
            self.docs.setdefault(path, {}).update(json.loads(json.dumps(fields, default=str)))

    def patch(self, path, mask, typed):
        self.check_online()
        with self.lock:
            doc = self.docs.setdefault(path, {})
            for name in mask:
                doc[name] = plain(typed[name])

    def upload(self, name, data, content_type="application/octet-stream"):
        self.check_online()
        path = self.root / "uploads" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.read_bytes() if isinstance(data, Path) else data)
        with self.lock:
            self.files[name] = path
            self.uploads.append(name)

    def delete(self, name):
        self.check_online()
        with self.lock:
            del self.files[name]
            self.deleted.append(name)


class TrainingTestCase(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.cloud = FakeTrainingCloud(self.tmp / "bucket")
        self.cloud.add_run(RUN, ["succeeded"] * 4, frames=2)
        self.source = dict(SOURCE)
        for name, value in [("TRAINING", self.tmp / "training"), ("SRC", self.tmp), ("FLUSH_SECONDS", 30),
                            ("RETRY_SECONDS", 0.01), ("GIVE_UP_SECONDS", 0.2),
                            ("source_of", lambda cloud, say: self.source),
                            ("where", lambda: {"kind": "laptop", "host": "kai-mac"})]:
            patcher = mock.patch.object(training, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Don't leave exit hooks and handlers behind.
        for patcher in (mock.patch.object(atexit, "register"), mock.patch.object(signal, "signal"),
                        mock.patch.object(sys, "excepthook", sys.excepthook)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write(f'runs = ["{RUN}"]\nvalidation = 0.25\n')
        self.runs: list[TrainingRun] = []
        self.addCleanup(self.stop_senders)

    def stop_senders(self):
        self.cloud.offline = False
        for run in self.runs:
            run._done = True
            run._stopping = True
            run._wake.set()
            run._sender.join(5)

    def make_model(self, name: str, train_toml: str | None = None) -> Path:
        folder = self.tmp / "training" / name
        folder.mkdir(parents=True, exist_ok=True)
        if train_toml is not None:
            (folder / "train.toml").write_text(train_toml)
        return folder

    def start(self, name="fuel_detector", resume=None) -> TrainingRun:
        run = TrainingRun.start(name, resume=resume, cloud=self.cloud, say=self.say)
        self.runs.append(run)
        return run

    def record(self, run) -> dict:
        return self.cloud.docs[f"training/{run if isinstance(run, str) else run.id}"]

    def checkpoint(self, run, step, data: bytes, **options) -> str:
        path = run.folder / "model.pt"
        path.write_bytes(data)
        return run.save(path, step, **options)


class TestStarting(TrainingTestCase):
    def test_the_record(self):
        self.make_model("fuel_detector", 'epochs = 3\nfreeze = true\n')
        run = self.start()

        record = self.record(run)
        self.assertEqual(record["model"], "fuel_detector")
        self.assertEqual(record["trained_by"], "kai-morales")
        self.assertEqual(record["status"], "running")
        self.assertEqual(record["where"], {"kind": "laptop", "host": "kai-mac"})
        self.assertEqual(record["segments"][0]["source"], SOURCE["source"])
        self.assertEqual(record["config"]["toml"], "epochs = 3\nfreeze = true\n")
        self.assertEqual(record["dataset"]["path"], "training/fuel_detector/dataset.toml")
        episodes = record["dataset"]["episodes"]
        self.assertEqual(sorted(episodes["train"] + episodes["validation"]), [f"{RUN}/{i}" for i in range(4)])
        self.assertEqual(sum(record["dataset"]["samples"].values()), 8)
        self.assertEqual(record["parent_runs"], [])
        self.assertEqual(run.config, {"epochs": 3, "freeze": True})
        self.assertIsNone(run.resume_from)
        self.assertTrue(any(line.startswith(f"Training run {run.id}: fuel_detector, by kai-morales") for line in self.said))

    def test_no_train_toml_is_fine(self):
        self.make_model("fuel_detector")
        run = self.start()
        self.assertEqual((run.config, self.record(run)["config"]["toml"]), ({}, None))

    def test_another_models_dataset(self):
        self.make_model("robot_vision", 'dataset = "../fuel_detector/dataset.toml"\n')
        self.assertEqual(self.record(self.start("robot_vision"))["dataset"]["path"],
                         "training/fuel_detector/dataset.toml")

    def test_mistakes(self):
        with self.assertRaisesRegex(ValueError, "isn't a model name"):
            self.start("Fuel-Detector")
        with self.assertRaisesRegex(ValueError, "There's no ai/src/training/nope/. The models are: fuel_detector"):
            self.start("nope")
        self.make_model("bad", "[start]\nbackbone = 3\n")
        with self.assertRaisesRegex(ValueError, r"\[start\] holds names"):
            self.start("bad")
        self.make_model("outside", 'dataset = "../../../elsewhere.toml"\n')
        with self.assertRaisesRegex(ValueError, "must be in ai/src"):
            self.start("outside")


class TestMetrics(TrainingTestCase):
    def test_sent_at_once_then_every_so_often(self):
        self.make_model("fuel_detector")
        run = self.start()
        run.log(0, loss=2.5, lr=0.01)
        run.log(1, loss=2.0, lr=0.01)  # within 30 s: held
        run._drain()
        self.assertEqual(self.record(run)["latest"], {"loss": 2.5, "lr": 0.01})

        run.flush()
        run._drain()
        files = sorted(n for n in self.cloud.files if "/metrics/" in n)
        self.assertEqual(files, [f"training/kai-morales/{run.id}/metrics/000000.jsonl",
                                 f"training/kai-morales/{run.id}/metrics/000001.jsonl"])
        rows = [json.loads(line) for line in self.cloud.files[files[1]].read_text().splitlines()]
        self.assertEqual([(r["step"], r["loss"]) for r in rows], [(1, 2.0)])
        self.assertEqual((self.record(run)["step"], self.record(run)["latest"]["loss"]), (1, 2.0))
        self.assertIn(f"fuel_detector {run.id} step 1: loss 2  lr 0.01", self.said)

    def test_numbers_only(self):
        self.make_model("fuel_detector")
        with self.assertRaisesRegex(TypeError, "loss must be a number, not str"):
            self.start().log(0, loss="low")


class TestCheckpoints(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector")

    def test_saving(self):
        run = self.start()
        run.log(100, loss=0.5)
        self.assertEqual(self.checkpoint(run, 100, b"first", metrics={"val_ap50": 0.5}), f"{run.id}/100")
        self.checkpoint(run, 200, b"better", resumable=True, metrics={"val_ap50": 0.7}, best=True)
        run._drain()

        name = f"training/kai-morales/{run.id}/checkpoints/200/model.pt"
        self.assertEqual(self.cloud.files[name].read_bytes(), b"better")
        checkpoint = self.cloud.docs[f"training/{run.id}/checkpoints/200"]
        self.assertEqual((checkpoint["file"], checkpoint["resumable"], checkpoint["best"], checkpoint["size"]),
                         (name, True, True, 6))
        record = self.record(run)
        self.assertEqual((record["last_checkpoint"], record["checkpoints"]), (200, 2))
        self.assertEqual(record["best"], {"step": 200, "metrics": {"val_ap50": 0.7}})
        self.assertEqual(jobs_in(run.queue), [])
        self.assertEqual([p.name for p in run.queue.iterdir()], [".lock"])  # copies deleted once sent

    def test_the_file_is_copied_so_it_can_be_overwritten_at_once(self):
        run = self.start()
        self.cloud.offline = True
        self.checkpoint(run, 1, b"step one")
        self.checkpoint(run, 2, b"step two")
        self.cloud.offline = False
        run._drain()
        self.assertEqual(self.cloud.files[f"training/kai-morales/{run.id}/checkpoints/1/model.pt"].read_bytes(),
                         b"step one")

    def test_no_such_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "no checkpoint file"):
            self.start().save(self.tmp / "missing.pt", 1)


class TestOffline(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector")

    def test_uploads_wait_and_go_in_order(self):
        run = self.start()
        self.cloud.offline = True
        run.log(0, loss=3.0)
        self.checkpoint(run, 10, b"weights")
        run.flush()
        self.assertEqual(len(jobs_in(run.queue)), 2)
        self.assertEqual(sum("Couldn't upload" in line for line in self.said), 1)  # said once

        self.cloud.offline = False
        run._drain()
        self.assertEqual(self.record(run)["last_checkpoint"], 10)
        metrics = [n for n in self.cloud.uploads if "/metrics/" in n]
        self.assertLess(self.cloud.uploads.index(metrics[0]),
                        self.cloud.uploads.index(f"training/kai-morales/{run.id}/checkpoints/10/model.pt"))

    def test_finishing_offline_leaves_them_for_sync(self):
        run = self.start()
        self.checkpoint(run, 10, b"weights")
        run._drain()
        self.cloud.offline = True
        self.checkpoint(run, 20, b"more weights")
        run.finish()

        self.assertEqual(self.record(run)["status"], "running")  # not yet
        self.assertTrue(any("run `wildhat training sync`" in line for line in self.said))
        queue = queue_folder(run.id)
        self.assertFalse((queue / ".lock").exists())  # so sync may send them

        self.cloud.offline = False
        for job in jobs_in(queue):  # what `wildhat training sync` does
            send_job(self.cloud, job)
        self.assertEqual((self.record(run)["status"], self.record(run)["last_checkpoint"]), ("finished", 20))


class TestPruning(TrainingTestCase):
    def test_keeps_the_newest_resumable_the_best_and_tagged(self):
        self.make_model("fuel_detector")
        run = self.start()
        self.checkpoint(run, 1, b"1", resumable=True)
        self.checkpoint(run, 2, b"2", resumable=True, best=True)
        self.checkpoint(run, 3, b"3", resumable=True)
        self.checkpoint(run, 10, b"weights only")
        run._drain()
        self.cloud.docs[f"training/{run.id}/checkpoints/3"]["tags"] = ["fuel_detector:v1"]
        for step in (4, 5, 6):
            self.checkpoint(run, step, str(step).encode(), resumable=True)
        run._drain()
        run.finish()

        pruned = {s for s in (1, 2, 3, 4, 5, 6, 10) if self.cloud.docs[f"training/{run.id}/checkpoints/{s}"]["pruned"]}
        self.assertEqual(pruned, {1})
        self.assertEqual(self.cloud.deleted, [f"training/kai-morales/{run.id}/checkpoints/1/model.pt"])
        with self.assertRaisesRegex(ValueError, "was deleted to save space"):
            training.resolve(self.cloud, f"{run.id}/1")


class TestStartingFromCheckpoints(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector")
        base = self.start()
        self.checkpoint(base, 100, b"early")
        self.checkpoint(base, 200, b"best", best=True)
        self.checkpoint(base, 300, b"last")
        base.finish()
        self.base = base.id

    def test_every_kind_of_reference(self):
        self.cloud.docs["models/fuel_detector"] = {"tags": {"v2": f"{self.base}/100"}}
        self.make_model("robot_vision", f'dataset = "../fuel_detector/dataset.toml"\n[start]\n'
                                        f'best = "{self.base}"\nstep = "{self.base}/300"\ntagged = "fuel_detector:v2"\n')
        run = self.start("robot_vision")

        record = self.record(run)
        self.assertEqual({k: v["checkpoint"] for k, v in record["parents"].items()},
                         {"best": f"{self.base}/200", "step": f"{self.base}/300", "tagged": f"{self.base}/100"})
        self.assertEqual(record["parents"]["tagged"]["ref"], "fuel_detector:v2")  # what was asked, and what it meant
        self.assertEqual(record["parents"]["best"]["model"], "fuel_detector")
        self.assertEqual(record["parent_runs"], [self.base])

        weights = run.start_weights()
        self.assertEqual({k: p.read_bytes() for k, p in weights.items()},
                         {"best": b"best", "step": b"last", "tagged": b"early"})
        count = len(self.cloud.downloads)
        run.start_weights()  # already here
        self.assertEqual(len(self.cloud.downloads), count)

    def test_a_run_without_a_best_means_its_last(self):
        del self.cloud.docs[f"training/{self.base}"]["best"]
        self.make_model("robot_vision", f'[start]\nw = "{self.base}"\n')
        shutil.copy(self.tmp / "training" / "fuel_detector" / "dataset.toml",
                    self.tmp / "training" / "robot_vision" / "dataset.toml")
        self.assertEqual(self.record(self.start("robot_vision"))["parents"]["w"]["checkpoint"], f"{self.base}/300")

    def test_bad_references(self):
        for ref, message in [("fuel_detector:v9", "No checkpoint is tagged fuel_detector:v9"),
                             ("20261231-2359-zzzz", "There's no training run 20261231-2359-zzzz"),
                             (f"{self.base}/250", "has no checkpoint at step 250"),
                             ("latest", "isn't a checkpoint")]:
            self.make_model("robot_vision", f'dataset = "../fuel_detector/dataset.toml"\n[start]\nw = "{ref}"\n')
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, message):
                self.start("robot_vision")


class TestResuming(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector", "epochs = 3\n")
        run = self.start()
        run.log(150, loss=0.4)
        self.checkpoint(run, 100, b"resume me 100", resumable=True)
        self.checkpoint(run, 200, b"resume me 200", resumable=True)
        self.checkpoint(run, 250, b"weights only")
        run.finish("stopped")
        self.first = run.id

    def test_carries_on_the_same_run(self):
        run = self.start(resume=self.first)

        self.assertEqual(run.id, self.first)
        self.assertEqual((run.resume_step, run.resume_from.read_bytes()), (200, b"resume me 200"))
        record = self.record(run)
        self.assertEqual((record["status"], record["finished_at"]), ("running", None))
        self.assertEqual(len(record["segments"]), 2)
        self.assertEqual(record["segments"][1]["resumed_at_step"], 200)
        self.assertFalse(any("has changed" in line for line in self.said))

        run.log(201, loss=0.3)
        self.checkpoint(run, 300, b"300", resumable=True, best=True)
        run.finish()
        metrics = sorted(n.rsplit("/", 1)[1] for n in self.cloud.files if "/metrics/" in n)
        self.assertEqual(metrics, ["000000.jsonl", "000001.jsonl"])  # numbering carries on
        self.assertEqual((self.record(run)["status"], self.record(run)["checkpoints"]), ("finished", 4))

    def test_says_what_changed(self):
        self.source = {**SOURCE, "source_sha256": "def"}
        (self.tmp / "training" / "fuel_detector" / "train.toml").write_text("epochs = 5\n")
        run = self.start(resume=self.first)
        self.assertEqual(run.config, {"epochs": 5})
        self.assertTrue(any("train.toml has changed" in line for line in self.said))
        self.assertTrue(any("The code has changed" in line for line in self.said))
        self.assertEqual([s["source_sha256"] for s in self.record(run)["segments"]], ["abc", "def"])

    def test_sends_whats_left_from_before_first(self):
        run = self.start(resume=self.first)
        self.cloud.offline = True
        self.checkpoint(run, 400, b"400", resumable=True)
        run.finish("stopped")
        self.cloud.offline = False

        again = self.start(resume=self.first)
        self.assertEqual(again.resume_step, 400)

    def test_mistakes(self):
        self.record(self.first)["trained_by"] = "dana-wells"
        with self.assertRaisesRegex(ValueError, "is dana-wells's, so only they can resume it"):
            self.start(resume=self.first)
        self.record(self.first)["trained_by"] = "kai-morales"

        self.record(self.first)["status"] = "running"
        self.record(self.first)["heartbeat_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.assertRaisesRegex(ValueError, "is still running"):
            self.start(resume=self.first)
        self.record(self.first)["status"] = "stopped"

        self.make_model("robot_vision", 'dataset = "../fuel_detector/dataset.toml"\n')
        with self.assertRaisesRegex(ValueError, "is fuel_detector, not robot_vision"):
            self.start("robot_vision", resume=self.first)

        for step in (100, 200):
            self.cloud.docs[f"training/{self.first}/checkpoints/{step}"]["resumable"] = False
        with self.assertRaisesRegex(ValueError, "has no resumable checkpoint"):
            self.start(resume=self.first)
        with self.assertRaisesRegex(ValueError, "isn't a training run"):
            self.start(resume="yesterday")


class TestEnding(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector")

    def test_with_block(self):
        with self.start() as run:
            run.log(1, loss=1.0)
        self.assertEqual(self.record(run)["status"], "finished")
        self.assertIsNotNone(self.record(run)["finished_at"])
        self.assertFalse(queue_folder(run.id).exists())

    def test_an_exception_fails_it(self):
        with self.assertRaises(ZeroDivisionError):
            with self.start() as run:
                1 / 0
        self.assertEqual((self.record(run)["status"], self.record(run)["error"]),
                         ("failed", "ZeroDivisionError: division by zero"))

    def test_ctrl_c_and_sigterm_stop_it(self):
        for error in (KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(error=error), self.assertRaises(type(error)):
                with self.start() as run:
                    raise error
            self.assertEqual(self.record(run)["status"], "stopped")

    def test_exit_zero_finishes_it(self):
        with self.assertRaises(SystemExit):
            with self.start() as run:
                sys.exit(0)
        self.assertEqual(self.record(run)["status"], "finished")

    def test_a_script_that_just_ends(self):
        run = self.start()
        (hook,) = [c.args[0] for c in atexit.register.call_args_list]
        hook()
        self.assertEqual(self.record(run)["status"], "finished")
        run.finish("failed")  # once finished, it stays finished
        self.assertEqual(self.record(run)["status"], "finished")

    def test_an_uncaught_exception_then_exit(self):
        run = self.start()
        try:
            raise ValueError("bad batch")
        except ValueError:
            with mock.patch("sys.stderr"):
                sys.excepthook(*sys.exc_info())
        (hook,) = [c.args[0] for c in atexit.register.call_args_list]
        hook()
        self.assertEqual((self.record(run)["status"], self.record(run)["error"]), ("failed", "ValueError: bad batch"))


if __name__ == "__main__":
    import unittest
    unittest.main()
