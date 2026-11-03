"""Training runs: the record, metrics, checkpoints, [start] references and
how a run ends. The cloud is faked (from test_datasets), and so is packing
the source, which `wildhat pack-source` does. Run from ai/src:

    python -m unittest discover -s shared/tests -t .
"""

import atexit
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from unittest import mock

from shared import training
from shared.tests.test_datasets import DatasetTestCase, FakeCloud
from shared.training import TrainingRun

RUN = "20261102-0019-w22f"
SOURCE = {"source": "gs://test-project-wildhat/src/kai-morales/abc.tar.gz", "source_sha256": "abc",
          "git": {"commit": "e86e8f6", "branch": "main", "dirty": True}}


class FakeTrainingCloud(FakeCloud):
    """FakeCloud, plus what training writes."""

    handle = "kai-morales"

    def __init__(self, root: Path):
        super().__init__(root)
        self.uploads: list[str] = []
        self.offline = False

    def create(self, path, fields):
        assert path not in self.docs, f"{path} exists"
        self.docs[path] = json.loads(json.dumps(fields, default=str))

    def update(self, path, fields):
        if self.offline:
            raise training.CloudError("Couldn't reach Google")
        self.docs.setdefault(path, {}).update(json.loads(json.dumps(fields, default=str)))

    def upload(self, name, data, content_type="application/octet-stream"):
        if self.offline:
            raise training.CloudError("Couldn't reach Google")
        path = self.root / "uploads" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.read_bytes() if isinstance(data, Path) else data)
        self.files[name] = path
        self.uploads.append(name)


class TrainingTestCase(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.cloud = FakeTrainingCloud(self.tmp / "bucket")
        self.cloud.add_run(RUN, ["succeeded"] * 4, frames=2)
        # Models live in a training directory of their own.
        for name, value in [("TRAINING", self.tmp / "training"), ("SRC", self.tmp), ("FLUSH_SECONDS", 30)]:
            patcher = mock.patch.object(training, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target in ("source_of", "where"):
            patcher = mock.patch.object(training, target, {
                "source_of": lambda cloud, say: SOURCE, "where": lambda: {"kind": "laptop", "host": "kai-mac"}}[target])
            patcher.start()
            self.addCleanup(patcher.stop)
        # Don't leave exit hooks and handlers behind.
        for patcher in (mock.patch.object(atexit, "register"), mock.patch.object(signal, "signal"),
                        mock.patch.object(sys, "excepthook", sys.excepthook)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write(f'runs = ["{RUN}"]\nvalidation = 0.25\n')

    def make_model(self, name: str, train_toml: str | None = None) -> Path:
        folder = self.tmp / "training" / name
        folder.mkdir(parents=True, exist_ok=True)
        if train_toml is not None:
            (folder / "train.toml").write_text(train_toml)
        return folder

    def start(self, name="fuel_detector") -> TrainingRun:
        return TrainingRun.start(name, self.cloud, self.say)

    def record(self, run: TrainingRun) -> dict:
        return self.cloud.docs[f"training/{run.id}"]

    def checkpoint_file(self, run: TrainingRun, name="model.pt", data=b"weights") -> Path:
        path = run.folder / name
        path.write_bytes(data)
        return path


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
        self.assertEqual(run.config, {"epochs": 3, "freeze": True})
        self.assertEqual(len(run.dataset.samples("train")) + len(run.dataset.samples("validation")), 8)
        self.assertTrue(any(line.startswith(f"Training run {run.id}: fuel_detector, by kai-morales") for line in self.said))

    def test_no_train_toml_is_fine(self):
        self.make_model("fuel_detector")
        run = self.start()
        self.assertEqual((run.config, self.record(run)["config"]["toml"]), ({}, None))

    def test_another_models_dataset(self):
        self.make_model("robot_vision", 'dataset = "../fuel_detector/dataset.toml"\n')
        run = self.start("robot_vision")
        self.assertEqual(self.record(run)["dataset"]["path"], "training/fuel_detector/dataset.toml")

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
        self.assertEqual(self.record(run)["latest"], {"loss": 2.5, "lr": 0.01})

        run.flush()
        files = sorted(n for n in self.cloud.files if "/metrics/" in n)
        self.assertEqual(files, [f"training/kai-morales/{run.id}/metrics/000000.jsonl",
                                 f"training/kai-morales/{run.id}/metrics/000001.jsonl"])
        rows = [json.loads(line) for line in self.cloud.files[files[1]].read_text().splitlines()]
        self.assertEqual([(r["step"], r["loss"]) for r in rows], [(1, 2.0)])
        self.assertEqual((self.record(run)["step"], self.record(run)["latest"]["loss"]), (1, 2.0))
        self.assertIn(f"fuel_detector {run.id} step 1: loss 2  lr 0.01", self.said)

    def test_numbers_only(self):
        self.make_model("fuel_detector")
        run = self.start()
        with self.assertRaisesRegex(TypeError, "loss must be a number, not str"):
            run.log(0, loss="low")

    def test_offline_metrics_wait_for_the_next_send(self):
        self.make_model("fuel_detector")
        run = self.start()
        self.cloud.offline = True
        run.log(0, loss=3.0)
        run.log(1, loss=2.0)
        run.flush()
        self.assertEqual(sum("Couldn't send metrics" in line for line in self.said), 1)  # said once
        self.cloud.offline = False
        run.flush()
        (name,) = [n for n in self.cloud.files if "/metrics/" in n]
        self.assertEqual(len(self.cloud.files[name].read_text().splitlines()), 2)


class TestCheckpoints(TrainingTestCase):
    def test_saving(self):
        self.make_model("fuel_detector")
        run = self.start()
        run.log(100, loss=0.5)
        self.assertEqual(run.save(self.checkpoint_file(run), 100, metrics={"val_ap50": 0.5}), f"{run.id}/100")
        run.save(self.checkpoint_file(run, data=b"better"), 200, resumable=True, metrics={"val_ap50": 0.7},
                 best=True)

        name = f"training/kai-morales/{run.id}/checkpoints/200/model.pt"
        self.assertEqual(self.cloud.files[name].read_bytes(), b"better")
        checkpoint = self.cloud.docs[f"training/{run.id}/checkpoints/200"]
        self.assertEqual((checkpoint["file"], checkpoint["resumable"], checkpoint["best"], checkpoint["size"]),
                         (name, True, True, 6))
        record = self.record(run)
        self.assertEqual((record["last_checkpoint"], record["checkpoints"]), (200, 2))
        self.assertEqual(record["best"], {"step": 200, "metrics": {"val_ap50": 0.7}})

    def test_no_such_file(self):
        self.make_model("fuel_detector")
        with self.assertRaisesRegex(FileNotFoundError, "no checkpoint file"):
            self.start().save(self.tmp / "missing.pt", 1)


class TestStartingFromCheckpoints(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector")
        base = self.start()
        base.save(self.checkpoint_file(base, data=b"early"), 100)
        base.save(self.checkpoint_file(base, data=b"best"), 200, best=True)
        base.save(self.checkpoint_file(base, data=b"last"), 300)
        base.finish()
        self.base = base.id

    def test_every_kind_of_reference(self):
        self.cloud.docs["models/fuel_detector"] = {"tags": {"v2": f"{self.base}/100"}}
        self.make_model("robot_vision", f'dataset = "../fuel_detector/dataset.toml"\n[start]\n'
                                   f'best = "{self.base}"\nstep = "{self.base}/300"\ntagged = "fuel_detector:v2"\n')
        run = self.start("robot_vision")

        parents = self.record(run)["parents"]
        self.assertEqual({k: v["checkpoint"] for k, v in parents.items()},
                         {"best": f"{self.base}/200", "step": f"{self.base}/300", "tagged": f"{self.base}/100"})
        self.assertEqual(parents["tagged"]["ref"], "fuel_detector:v2")  # what was asked, and what it meant
        self.assertEqual(parents["best"]["model"], "fuel_detector")

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


class TestEnding(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.make_model("fuel_detector")

    def test_with_block(self):
        with self.start() as run:
            run.log(1, loss=1.0)
        self.assertEqual(self.record(run)["status"], "finished")
        self.assertIsNotNone(self.record(run)["finished_at"])

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
