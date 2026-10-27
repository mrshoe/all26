# wildhat

The command-line tool for Team 100's Wildhat AI system. It signs you in to the
team's Google Cloud project, so that `gcloud` and the Google Cloud Python
libraries work as you, and it runs your code in the cloud.

## Install

Install the Google Cloud CLI and uv. You don't need Python; uv brings its own.

```
winget install --id Google.CloudSDK -e      # Windows (PowerShell)
winget install --id astral-sh.uv -e

brew install --cask gcloud-cli              # macOS
brew install uv
```

Then, from the root of this repo:

```
uv tool install ./ai/wildhat
uv tool update-shell
```

Open a new terminal so `wildhat` is on your PATH, and check with `wildhat
--version`.

### Upgrade

From the root of the repo, get the latest code and reinstall:

```
git pull
uv tool install --reinstall ./ai/wildhat
```

`--reinstall` is needed: without it, uv keeps the version you already have.
You stay signed in. `wildhat --version` shows which version you have.

## Sign in

```
wildhat login
```

Type your school email. You'll get an email with a sign-in link. Open your
school Gmail ([gmail.com](https://gmail.com)) in a browser **on the same
computer** and click the link; sign-in finishes by itself. Clicking it on your
phone won't work. (If you really can't open the email on this computer, you
can paste the whole link into the terminal instead.)

At the end, login sets two environment variables so that `gcloud` and the
Google Cloud Python libraries work as you in any new terminal. On Windows it
just sets them; on macOS it asks before adding them to `~/.zshrc`. Then open
a new terminal and check:

```
wildhat whoami
gcloud run jobs list
```

You only need to sign in once. Your sign-in lasts until a mentor removes your
access.

## Run something in the cloud

The code that runs in the cloud lives in `ai/src` (see
[../src/README.md](../src/README.md)). From anywhere inside it:

```
wildhat launch -- python train.py --epochs 25
wildhat launch --gpu -- python train.py
wildhat launch --episodes 50 --timeout 2h -- python sim.py --scenario corner
```

launch packs up `ai/src` exactly as it is on your computer, uncommitted
changes and new files included, uploads it, and starts the run. Nothing
outside `ai/src` is uploaded. It
prints the run's ID (like `20261026-1430-ab12`) and returns straight away;
the run carries on in the cloud.

| Option | What it does |
|---|---|
| `--episodes N` | Runs N copies of the command, each with its own seed (default 1, at most 100). Use it for simulations, data generation and evals. Training runs leave it out. |
| `--gpu` | Gives each episode an NVIDIA L4. Not available until the project gets GPU quota. |
| `--timeout 90m` | How long each episode may run (default 1h, at most 8h). |
| `--seed S` | Replays one episode with the seed it had (needs `--episodes 1`). |

Your command can read:

- `WILDHAT_EPISODE`: this episode's number (0, 1, 2, ...)
- `WILDHAT_EPISODE_COUNT`: how many episodes the run has
- `WILDHAT_SEED`: this episode's random seed. Use it for everything random,
  so the episode can be replayed.
- `WILDHAT_OUTPUT_DIR`: a folder for results. Everything written there is
  saved with the run when the episode ends, even if it fails, times out or is
  cancelled. Get it with `wildhat download`.

When an episode times out or is cancelled, your command gets a `SIGTERM`
first. Python stops at once by default, without running `finally:` blocks, so
to save a checkpoint on the way out, handle it:

```python
import signal, sys
def stop(signum, frame):
    save_checkpoint()  # quickly: there are only a few seconds
    sys.exit(1)
signal.signal(signal.SIGTERM, stop)
```

The command runs in the same folder you launched from. Files that `.gitignore`
ignores aren't uploaded, and the upload can be at most 100 MB, so keep
datasets in the bucket. If `ai/src/requirements.txt` exists, it's installed
before each episode starts; the cloud already has Python 3.11, PyTorch (with
CUDA) and ultralytics.

**Everyone on the team can read what launch uploads.** It names any files git
doesn't track yet, so you can spot a private note before it goes. Move those
out of `ai/src`, or add them to `.gitignore`.

Each run keeps a record of who launched it, the command, the git commit it
started from (and whether there were uncommitted changes), and each episode's
seed, status and outputs; `wildhat runs RUN` shows it. The exact code it ran
is kept too, so `wildhat download RUN --source` always gets what a run
really ran.

The first run after the cloud image changes waits about 10 minutes while
Cloud Run loads the image; after that runs start in seconds.

## Follow your runs

```
wildhat runs                       # everyone's latest runs
wildhat runs --mine                # yours (or --user jane-doe for someone's)
wildhat runs 20261026-1430-ab12    # one run: its code, command, and each episode
wildhat logs 20261026-1430-ab12 -f # its output, as it happens
wildhat cancel 20261026-1430-ab12
wildhat download 20261026-1430-ab12
```

| Command | What it does |
|---|---|
| `wildhat runs [RUN]` | Lists the latest runs (everyone's; `--mine`, `--user HANDLE`, `--limit N`), or shows one: the command, the git commit it started from, and each episode's seed, status, exit code and time. |
| `wildhat logs RUN` | Prints a run's output so far, each line marked with its episode, like `[3]`. `-f` keeps printing until the run ends; `--episode N` shows one episode. Logs are kept for 30 days. |
| `wildhat cancel RUN` | Cancels one of your runs, after asking (`--yes` doesn't ask). Running episodes get a `SIGTERM` and a few seconds to save their outputs. |
| `wildhat download RUN` | Downloads a run's outputs to `./RUN/episode-<n>/` (`--episode N` for one, `--to DIR` for somewhere else). `--source` downloads the exact code it ran instead. Not into `ai/src`, which the next launch would upload. |

You can see, follow and download everyone's runs, but only cancel your own.
A run's status is `starting`, `running (2 of 5 done)`, `succeeded`,
`failed (1 of 5)` or `cancelled`; an episode that was cancelled before it
started shows as `not started`, and one that timed out or was cancelled
while running as `stopped`.

## Other commands

| Command | What it does |
|---|---|
| `wildhat whoami` | Shows your email, your handle, and whether your access works right now. |
| `wildhat logout` | Signs you out and deletes your saved credentials. |
| `wildhat id-token` | Used by gcloud behind the scenes. You never need to run it. |

## For mentors: the roster

The roster (each student's handle, school email and name) is private, so it
lives in a private bucket, `gs://vernal-mantra-512116-d0-admin/roster.json`,
and never in this repo. Sign in with your own account first (`gcloud auth
login`), then:

| Command | What it does |
|---|---|
| `wildhat admin edit-roster` | Opens the roster in `$EDITOR` (Notepad on Windows if it's not set), checks it, shows your changes, saves it, syncs Firebase, then shows Terraform's plan and offers to apply it. |
| `wildhat admin sync [--dry-run]` | Makes Firebase match the roster. Safe to run any time. |
| `wildhat admin apply` | Shows Terraform's plan for the roster and offers to apply it. |
| `wildhat admin cancel-jobs [--running-longer-than MINUTES]` | Kill switch: cancels every running Cloud Run job execution (or only those running longer than MINUTES). |

Run these from inside your all26 checkout so wildhat can find `ai/terraform`.
You also need Terraform: `winget install --id Hashicorp.Terraform -e` or
`brew install hashicorp/tap/terraform`.

Sync creates a Firebase user for each new student and sets their `handle`
claim and name. Anyone who has a handle but is no longer on the roster is
disabled, which cuts off their access immediately. **To revoke a student's
access, delete their entry.**

Sync handles sign-in. Each student's service account and IAM grants come
from Terraform, which reads the same `roster.json`, so there's only one
roster.

### Mentors are on the roster too

Add yourself with your team100.org address and run `wildhat login` like a
student. Your jobs then run as your own `wh-<handle>` service account, with
exactly a student's access, so they look and behave just like students' jobs.

Your Google account stays gcloud's active account: `wildhat login` doesn't
switch it if a Google account is already active. The admin commands always
use your Google account and everything else in wildhat uses your service
account, whichever one is active. To run gcloud yourself as the service
account, add `--account=wh-<handle>@vernal-mantra-512116-d0.iam.gserviceaccount.com`.

## How it works

You sign in to Firebase Authentication with an email link. Google Cloud's
workload identity federation trusts Firebase's ID tokens, so gcloud trades
your token for a short-lived Google access token and acts as your own service
account, `wh-<first-last>`. Whenever gcloud needs a fresh token, it runs
`wildhat id-token`.

Your credentials are stored in `~/.config/wildhat/` (macOS and Linux) or
`%APPDATA%\wildhat\` (Windows). Don't share the files in that folder.

## Tests

From this directory (nothing to install):

```
uv run --no-project --python 3.13 python -m unittest discover -s tests
```

The tests run the real code in `src/` through its command line. They fake
only what's outside wildhat: Google's web APIs, the keyboard, and the `gcloud`,
`terraform` and editor programs (`tests/fake_tool.py`). The fake gcloud runs
`wildhat id-token` from the credential config the way Google's libraries do,
from a folder with a space in its name. GitHub runs the tests on macOS,
Windows and Linux whenever `ai/wildhat/` changes
(`.github/workflows/wildhat.yml`). The same workflow runs the job watchdog's
tests, in `ai/terraform/functions/tests/` (run them from
`ai/terraform/functions` the same way).
