# wildhat

The command-line tool for Team 100's Wildhat AI system. For now it does one
thing: it signs you in to the team's Google Cloud project, so that `gcloud`
and the Google Cloud Python libraries work as you.

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
