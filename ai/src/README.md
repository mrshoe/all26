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
- `requirements.txt` (optional): extra Python packages, installed before each
  episode starts. The cloud image already has Python 3.11, PyTorch with CUDA,
  and ultralytics.

Launch from the folder your command expects to run in, e.g.
`cd ai/src/simulator` then `wildhat launch -- python run.py`.

Everyone on the team can read what launch uploads. Keep private notes out of
this folder, or in `.gitignore`. Datasets go in the bucket, not here: an upload
can be at most 100 MB.
