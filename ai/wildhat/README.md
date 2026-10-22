# wildhat

The command-line tool for Team 100's Wildhat AI system. For now it does one
thing: it signs you in to the team's Google Cloud project, so that `gcloud`
and the Google Cloud Python libraries work as you.

## Install

You need the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the
root of this repo:

```
uv tool install ./ai/wildhat
uv tool update-shell
```

Open a new terminal afterwards so `wildhat` is on your PATH. (`pipx install
./ai/wildhat` works too.)

## Sign in

```
wildhat login
```

Type your school email. You'll get an email with a sign-in link. Open your
school Gmail ([gmail.com](https://gmail.com)) in a browser **on the same
computer** and click the link; sign-in finishes by itself. Clicking it on your
phone won't work. (If you really can't open the email on this computer, you
can paste the whole link into the terminal instead.)

You only need to do this once. Your sign-in lasts until a mentor removes your
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
| `wildhat admin edit-roster` | Opens the roster in `$EDITOR`, checks it, shows your changes, saves it, syncs Firebase, then shows Terraform's plan and offers to apply it. |
| `wildhat admin sync [--dry-run]` | Makes Firebase match the roster. Safe to run any time. |
| `wildhat admin apply` | Shows Terraform's plan for the roster and offers to apply it. |
| `wildhat admin cancel-jobs [--running-longer-than MINUTES]` | Kill switch: cancels every running Cloud Run job execution (or only those running longer than MINUTES). |

Run these from inside your all26 checkout so wildhat can find `ai/terraform`.
You also need [Terraform](https://developer.hashicorp.com/terraform/install).

Sync creates a Firebase user for each new student and sets their `handle`
claim and name. Anyone who has a handle but is no longer on the roster is
disabled, which cuts off their access immediately. **To revoke a student's
access, delete their entry.**

Sync handles sign-in. Each student's service account and IAM grants come
from Terraform, which reads the same `roster.json`, so there's only one
roster.

## How it works

You sign in to Firebase Authentication with an email link. Google Cloud's
workload identity federation trusts Firebase's ID tokens, so gcloud trades
your token for a short-lived Google access token and acts as your own service
account, `stu-<first-last>`. Whenever gcloud needs a fresh token, it runs
`wildhat id-token`.

Your credentials are stored in `~/.config/wildhat/` (macOS) or
`%APPDATA%\wildhat\` (Windows). Don't share the files in that folder.
