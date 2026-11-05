# The visualizer

A web app for looking at runs: everyone's runs, newest first, and for
simulation episodes, each camera's video (with or without its labels:
boxes, full boxes and keypoints) next to
plots of the episode's signals, with a playhead that follows the video.

And training runs (`ai/src/shared/training.py` records them): the list,
filtered by model or person; each run's code, dataset, the checkpoints it
started from and the runs that started from it, a table of its metrics
by split (train, val, test: the names' prefixes) with each one's best,
a plot of each metric against step with its splits together (refreshed
every 30 seconds while it runs), the pictures and files it kept, and its
checkpoints; and several runs' metrics overlaid, to compare them. A cloud
job that ran a training run links to it from the Runs page.

Sign in with the email address on the team roster, the same way as
`wildhat login`. The address is printed by `terraform output visualizer_url`
in `ai/terraform`.

```
server/app.py   the API (FastAPI) and the built app
web/            the browser app (TypeScript, Vite, uPlot)
Dockerfile      builds both into one image
```

It's one Cloud Run service with no instances while nobody's using it
([../INFRA.md](../INFRA.md)), deployed by `wildhat admin apply`
([../terraform/visualizer.tf](../terraform/visualizer.tf)). The server reads
Firestore and the bucket as its own service account, which can't change
anything. Videos play straight from the bucket through signed links.

## Working on it

The server, with Python 3.13 (it reads the real project's data, as you):

```sh
cd ai/visualizer/server
uv venv --python 3.13 && uv pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests
PROJECT_ID=vernal-mantra-512116-d0 FIREBASE_API_KEY=... \
  .venv/bin/uvicorn app:main --factory --port 8080
```

Signed links need a service account to sign as, so videos only play when
it runs in Cloud Run.

The app, with Node 22:

```sh
cd ai/visualizer/web
npm install
npm run dev     # http://localhost:5173, passing /api to the server on :8080
npm run check   # type-check
```
