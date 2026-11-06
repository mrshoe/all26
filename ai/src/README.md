# ai/src: the code that runs in the cloud

`wildhat launch` uploads this folder, exactly as it is on your computer, and
runs your command in a copy of it (see [../wildhat/README.md](../wildhat/README.md)).
Nothing outside it is uploaded, so plans, notes, docs and tooling stay out.

- `simulator/`: the simulator: Scenes, Scenarios, physics and rendering.
- `training/`: supervised and RL training, and evals.
- `shared/`: code the others share, and the robot too: the `Logger` and
  `PostProcessor` ([../README.md](../README.md)), and `episodes.py`, which
  reads what the `Logger` writes. Import it as `from shared.logger import
  Logger`, with `ai/src` on the path (`sim` puts it there).
  - `datasets.py`: a model's `dataset.toml` (the runs it trains on, and how
    they split), downloading those runs, and its samples, or all of it in
    ultralytics YOLO's format (`dataset.yolo(classes=["ball"])`). Try
    `python -m shared.datasets check training/<model>` from here.
  - `training.py`: `TrainingRun`, which records every training run in the
    `training` database, wherever it runs, and uploads its metrics,
    checkpoints, final results and kept files (plots, pictures) in the
    background for anyone to build on. A model's
    `train.toml` names its dataset and the checkpoints it starts from; a run
    that stopped can be resumed. `wildhat training` lists, shows, tags and
    downloads them. Needs wildhat 0.6.0 on your computer.
  - `predictions.py` and `score.py`: scoring a model the same way for
    everyone. Write its boxes and keypoints on a split's pictures with
    `Predictions(folder, model="fuel_detector", classes=["fuel"])`, then
    `python -m shared.score <folder>` prints AP, AP50 and AP75 per class and
    by object size, the score threshold with the best F1, and for keypoints
    the share found within 2, 4 and 8 pixels, by seen and hidden. Objects
    under `--min-pixels` or `--min-share` are ignored, not misses.
    `metrics(score(...), "test")` gives flat numbers for `run.final(...)`.
  - `cloud.py`: reading and writing the bucket and Firestore, as you, the
    same way on your computer (through `wildhat login`; needs wildhat 0.4.0)
    and in a cloud job.
- `requirements.txt` (optional): extra Python packages, installed before each
  episode starts. The cloud image already has Python 3.11, PyTorch with CUDA,
  and ultralytics.

Launch from the folder your command expects to run in, e.g.
`cd ai/src/simulator` then `wildhat launch -- python run.py`.

Everyone on the team can read what launch uploads. Keep private notes out of
this folder, or in `.gitignore`. Datasets go in the bucket, not here: an upload
can be at most 100 MB.
