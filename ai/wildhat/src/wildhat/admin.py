"""Mentor-only commands: edit the team roster, sync it to Firebase, and apply
the matching Terraform.

These run with the mentor's own gcloud credentials. Students can run them too,
but Google will refuse every call, because their service accounts have no
access to the admin bucket or to Firebase user management.

The roster's master copy is one JSON object in a private bucket:

    {
      "students": {
        "kai-morales": {"email": "813579@seq.org", "name": "Kai Morales"}
      }
    }

The roster is the source of truth. Sync makes Firebase match it: every student
gets a Firebase user whose `handle` custom claim is their key in the roster,
and anyone with a handle who is no longer on the roster is disabled. To revoke
a student's access, remove them from the roster.

Terraform (ai/terraform) reads the same roster object to create each
student's service account and IAM grants. edit-roster saves, syncs, then shows
Terraform's plan and offers to apply it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from wildhat.cli import (
    HANDLE_MAX,
    HANDLE_RE,
    IDENTITY_TOOLKIT,
    IS_WINDOWS,
    PROJECT_ID,
    REGION,
    WildhatError,
    gcloud,
    gcloud_accounts,
    is_service_account,
)

ADMIN_BUCKET = f"{PROJECT_ID}-admin"
ROSTER_OBJECT = "roster.json"
ROSTER_URI = f"gs://{ADMIN_BUCKET}/{ROSTER_OBJECT}"
STORAGE = "https://storage.googleapis.com"
ACCOUNTS = f"{IDENTITY_TOOLKIT}/projects/{PROJECT_ID}/accounts"

ROSTER_HELP = """\
Roster format: one entry per student, keyed by handle (first-last, lowercase
letters and hyphens, a trailing number for a second Jane Doe: jane-doe2).

  "students": {
    "jane-doe": {"email": "123456@seq.org", "name": "Jane Doe"}
  }

To revoke a student's access, delete their entry.
"""


# ---------------------------------------------------------------------------
# Authenticated Google API calls, as the mentor
# ---------------------------------------------------------------------------


def mentor_account() -> str:
    """The mentor's own Google account in gcloud.

    A mentor who has also run `wildhat login` has their wh-<handle> service
    account signed in too, so don't rely on whichever account is active.
    """
    users = [a for a in gcloud_accounts() if not is_service_account(a["account"])]
    for a in users:
        if a["status"] == "ACTIVE":
            return a["account"]
    if len(users) == 1:
        return users[0]["account"]
    if not users:
        raise WildhatError(
            "Admin commands need a mentor's Google account, and none is signed "
            "in to gcloud. Run: gcloud auth login"
        )
    raise WildhatError(
        "Several Google accounts are signed in to gcloud: "
        + ", ".join(a["account"] for a in users)
        + ". Pick yours with: gcloud config set account EMAIL"
    )


def access_token(account: str) -> str:
    result = gcloud("auth", "print-access-token", f"--account={account}", check=False)
    if result.returncode != 0:
        raise WildhatError(
            f"Couldn't get gcloud credentials for {account}. Run: gcloud auth login\n"
            + result.stderr.strip()
        )
    return result.stdout.strip()


class Api:
    def __init__(self):
        self.account = mentor_account()
        self.token = access_token(self.account)

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            # User credentials need a project to bill API quota to.
            "x-goog-user-project": PROJECT_ID,
        }
        if body is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, error_message(e)) from None
        except urllib.error.URLError as e:
            raise WildhatError(f"Couldn't reach Google ({e.reason}). Are you online?")

    def json(self, method: str, url: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        data, _ = self.request(method, url, body)
        return json.loads(data) if data else {}


class ApiError(WildhatError):
    def __init__(self, status: int, message: str):
        self.status = status
        hint = ""
        if status in (401, 403):
            hint = (
                "\nAdmin commands need a mentor's Google account. Check which "
                "accounts gcloud has with: gcloud auth list"
            )
        super().__init__(f"Google said (HTTP {status}): {message}{hint}")


def error_message(e: urllib.error.HTTPError) -> str:
    try:
        return json.load(e)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return e.reason


# ---------------------------------------------------------------------------
# The roster file
# ---------------------------------------------------------------------------


def download_roster(api: Api) -> tuple[str, int]:
    """Return the roster's text and generation (0 if it doesn't exist yet)."""
    url = (
        f"{STORAGE}/storage/v1/b/{ADMIN_BUCKET}/o/"
        f"{urllib.parse.quote(ROSTER_OBJECT, safe='')}?alt=media"
    )
    try:
        data, headers = api.request("GET", url)
    except ApiError as e:
        if e.status == 404:
            return json.dumps({"students": {}}), 0
        raise
    generation = {k.lower(): v for k, v in headers.items()}["x-goog-generation"]
    return data.decode("utf-8"), int(generation)


def upload_roster(api: Api, roster: dict, generation: int) -> None:
    """Save the roster, but only if nobody else has changed it meanwhile."""
    query = urllib.parse.urlencode(
        {"uploadType": "media", "name": ROSTER_OBJECT, "ifGenerationMatch": generation}
    )
    url = f"{STORAGE}/upload/storage/v1/b/{ADMIN_BUCKET}/o?{query}"
    try:
        api.request("POST", url, format_roster(roster).encode("utf-8"))
    except ApiError as e:
        if e.status == 412:
            raise WildhatError(
                "Someone else changed the roster while you were editing it. "
                "Run edit-roster again."
            )
        raise


def format_roster(roster: dict) -> str:
    return json.dumps(roster, indent=2, sort_keys=True) + "\n"


def parse_roster(text: str) -> dict:
    """Parse and check a roster. Raises WildhatError explaining what's wrong."""

    def no_duplicates(pairs):
        keys = [k for k, _ in pairs]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise WildhatError(f"Listed more than once: {', '.join(dupes)}")
        return dict(pairs)

    try:
        roster = json.loads(text, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as e:
        raise WildhatError(f"Not valid JSON: {e}")
    if not isinstance(roster, dict) or not isinstance(roster.get("students"), dict):
        raise WildhatError('The roster must be an object with a "students" object.')

    problems = []
    emails = {}
    for handle, student in roster["students"].items():
        if not HANDLE_RE.match(handle) or len(handle) > HANDLE_MAX:
            problems.append(
                f"{handle!r}: handles are first-last, lowercase letters and "
                f"hyphens, optional trailing number, at most {HANDLE_MAX} characters"
            )
        if not isinstance(student, dict) or set(student) != {"email", "name"}:
            problems.append(f"{handle}: needs exactly \"email\" and \"name\"")
            continue
        email, name = student["email"], student["name"]
        if not isinstance(email, str) or "@" not in email or email != email.strip().lower():
            problems.append(f"{handle}: email must be a lowercase address")
        elif email in emails:
            problems.append(f"{handle}: same email as {emails[email]}")
        else:
            emails[email] = handle
        if not isinstance(name, str) or not name.strip():
            problems.append(f"{handle}: name is missing")
    if problems:
        raise WildhatError("Problems in the roster:\n  " + "\n  ".join(problems))
    return roster


def describe_changes(old: dict, new: dict) -> list[str]:
    old, new = old["students"], new["students"]
    lines = [f"  + {h} ({new[h]['email']})" for h in sorted(new.keys() - old.keys())]
    lines += [f"  - {h} ({old[h]['email']})" for h in sorted(old.keys() - new.keys())]
    for h in sorted(old.keys() & new.keys()):
        for field in ("email", "name"):
            if old[h][field] != new[h][field]:
                lines.append(f"  ~ {h}: {field} {old[h][field]!r} -> {new[h][field]!r}")
    return lines


def editor_command() -> str:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        return editor
    if IS_WINDOWS:
        return "notepad"
    return "nano" if shutil.which("nano") else "vi"


def run_editor(path: Path) -> None:
    # Through the shell, so EDITOR can carry its own flags ("code --wait").
    result = subprocess.run(f'{editor_command()} "{path}"', shell=True)
    if result.returncode != 0:
        raise WildhatError(f"The editor exited with an error. Your edits are in {path}")


def confirm(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Sync: make Firebase Authentication match the roster
# ---------------------------------------------------------------------------


def list_users(api: Api) -> list[dict]:
    users, page = [], ""
    while True:
        query = urllib.parse.urlencode({"maxResults": 1000, "nextPageToken": page})
        reply = api.json("GET", f"{ACCOUNTS}:batchGet?{query}")
        users += reply.get("users", [])
        page = reply.get("nextPageToken")
        if not page:
            return users


def claims_of(user: dict) -> dict:
    try:
        return json.loads(user.get("customAttributes") or "{}")
    except ValueError:
        return {}


def sync(api: Api, roster: dict, dry_run: bool = False) -> None:
    users = {u["email"].lower(): u for u in list_users(api) if u.get("email")}
    students = roster["students"]
    on_roster = {s["email"] for s in students.values()}
    unchanged = 0

    def act(message: str, method: str, url: str, payload: dict) -> dict:
        print(("would " if dry_run else "") + message)
        return {} if dry_run else api.json(method, url, payload)

    for handle, student in sorted(students.items()):
        email, name = student["email"], student["name"]
        user = users.get(email)
        if user is None:
            reply = act(
                f"create {handle} ({email})", "POST", ACCOUNTS,
                {"email": email, "displayName": name},
            )
            user = {"localId": reply.get("localId")}
        claims = claims_of(user)
        if (
            claims.get("handle") == handle
            and user.get("displayName") == name
            and not user.get("disabled")
        ):
            unchanged += 1
            continue
        claims["handle"] = handle
        act(
            f"set {handle}: handle claim, name {name!r}, enabled", "POST",
            f"{ACCOUNTS}:update",
            {
                "localId": user["localId"],
                "displayName": name,
                "customAttributes": json.dumps(claims),
                "disableUser": False,
            },
        )

    # Anyone who had a handle but is no longer on the roster loses it, and is
    # disabled, which also cancels their refresh token right away.
    for email, user in sorted(users.items()):
        claims = claims_of(user)
        if "handle" in claims and email not in on_roster:
            old = claims.pop("handle")
            act(
                f"disable {old} ({email}): not on the roster", "POST",
                f"{ACCOUNTS}:update",
                {
                    "localId": user["localId"],
                    "customAttributes": json.dumps(claims),
                    "disableUser": True,
                },
            )
    print(f"{unchanged} student(s) already up to date.")


# ---------------------------------------------------------------------------
# Terraform: each student's service account, IAM grants, and so on
# ---------------------------------------------------------------------------

TERRAFORM_SUBDIR = Path("ai") / "terraform"


def terraform_dir() -> Path:
    """Find ai/terraform in the all26 checkout the mentor is working in."""
    override = os.environ.get("WILDHAT_TERRAFORM_DIR")
    if override:
        return Path(override)
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / TERRAFORM_SUBDIR / "main.tf").exists():
            return d / TERRAFORM_SUBDIR
    raise WildhatError(
        f"Couldn't find {TERRAFORM_SUBDIR}. Run this from inside your all26 "
        "checkout (or set WILDHAT_TERRAFORM_DIR)."
    )


def plan_and_apply(api: Api) -> int:
    """Show Terraform's plan, and apply it if the mentor agrees."""
    tf_dir = terraform_dir()
    terraform = shutil.which("terraform")
    if not terraform:
        raise WildhatError(
            "Couldn't find terraform. Install it: "
            "https://developer.hashicorp.com/terraform/install"
        )
    # The same credentials as the rest of wildhat admin, so mentors don't
    # also need `gcloud auth application-default login`.
    env = dict(os.environ, GOOGLE_OAUTH_ACCESS_TOKEN=api.token)

    def tf(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
        sys.stdout.flush()  # So our messages land before Terraform's.
        return subprocess.run(
            [terraform, f"-chdir={tf_dir}", *args],
            env=env,
            capture_output=capture,
            text=True,
        )

    init = tf("init", "-input=false", capture=True)
    if init.returncode != 0:
        raise WildhatError(f"terraform init failed:\n{init.stdout}{init.stderr}")

    # The plan file can hold student emails, so keep it out of the repo.
    with tempfile.TemporaryDirectory() as tmp:
        plan_file = Path(tmp) / "roster.tfplan"
        print("Planning cloud changes with Terraform...\n")
        plan = tf("plan", "-input=false", "-detailed-exitcode", f"-out={plan_file}")
        if plan.returncode == 0:
            print("Cloud resources already match the roster.")
            return 0
        if plan.returncode != 2:
            raise WildhatError("terraform plan failed; see above.")
        if not confirm("\nApply this plan?"):
            print("Not applied. Run `wildhat admin apply` when you're ready.")
            return 1
        if tf("apply", "-input=false", str(plan_file)).returncode != 0:
            raise WildhatError("terraform apply failed; see above.")
    return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_edit_roster(args) -> int:
    api = Api()
    text, generation = download_roster(api)
    old = parse_roster(text) if generation else {"students": {}}

    fd, name = tempfile.mkstemp(prefix="roster-", suffix=".json")
    path = Path(name)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(format_roster(old))

    print(ROSTER_HELP)
    print(f"Opening {ROSTER_URI} in {editor_command()}...")
    while True:
        run_editor(path)
        try:
            new = parse_roster(path.read_text(encoding="utf-8"))
            break
        except WildhatError as e:
            print(e)
            if not confirm("Edit again?"):
                print(f"Nothing saved. Your edits are in {path}")
                return 1

    changes = describe_changes(old, new)
    if not changes:
        path.unlink()
        print("No changes.")
        return 0
    print("\nChanges:")
    print("\n".join(changes))
    if not confirm("\nSave the roster and sync it to Firebase?"):
        print(f"Nothing saved. Your edits are in {path}")
        return 1
    try:
        upload_roster(api, new, generation)
    except WildhatError as e:
        raise WildhatError(f"{e}\nYour edits are in {path}")
    path.unlink()
    print(f"Saved {ROSTER_URI}.\n")
    sync(api, new)
    print()
    return plan_and_apply(api)


def cmd_sync(args) -> int:
    api = Api()
    text, generation = download_roster(api)
    if not generation:
        raise WildhatError(f"There's no roster at {ROSTER_URI} yet. Run edit-roster.")
    sync(api, parse_roster(text), dry_run=args.dry_run)
    return 0


def cmd_apply(args) -> int:
    return plan_and_apply(Api())


# ---------------------------------------------------------------------------
# Kill switch: the job watchdog function (ai/terraform/guardrails.tf)
# ---------------------------------------------------------------------------

WATCHDOG = (
    f"https://cloudfunctions.googleapis.com/v2/projects/{PROJECT_ID}"
    f"/locations/{REGION}/functions/job-watchdog"
)


def cmd_cancel_jobs(args) -> int:
    minutes = args.running_longer_than
    which = (
        "EVERY running job execution"
        if minutes == 0
        else f"every job execution running longer than {minutes} minutes"
    )
    if not args.yes and not confirm(f"Cancel {which}?"):
        return 1

    api = Api()
    url = api.json("GET", WATCHDOG)["serviceConfig"]["uri"]
    # The function only accepts Google-signed identity tokens.
    token = gcloud("auth", "print-identity-token", f"--account={api.account}").stdout.strip()
    req = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode({'max_minutes': minutes})}",
        data=b"",
        headers={"Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.load(resp)
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.reason) from None

    if result.get("month_cost") is not None:
        print(f"This month's cost so far: ${result['month_cost']:.2f}")
    print(f"Checked {result['checked']} execution(s).")
    for name in result["cancelled"]:
        print(f"  cancelled {name.split('/jobs/')[-1]}")
    if not result["cancelled"]:
        print("Nothing to cancel.")
    return 0


def add_parsers(sub) -> None:
    admin = sub.add_parser("admin", help="mentor-only commands")
    admin_sub = admin.add_subparsers(dest="admin_command", required=True)
    admin_sub.add_parser(
        "edit-roster", help="edit the team roster, then sync it to Firebase"
    ).set_defaults(func=cmd_edit_roster)
    sync_parser = admin_sub.add_parser("sync", help="make Firebase match the roster")
    sync_parser.add_argument(
        "--dry-run", action="store_true", help="show what would change"
    )
    sync_parser.set_defaults(func=cmd_sync)
    admin_sub.add_parser(
        "apply", help="show Terraform's plan for the roster and offer to apply it"
    ).set_defaults(func=cmd_apply)
    cancel = admin_sub.add_parser(
        "cancel-jobs", help="kill switch: cancel running Cloud Run job executions"
    )
    cancel.add_argument(
        "--running-longer-than",
        type=float,
        default=0,
        metavar="MINUTES",
        help="only cancel executions running longer than this (default: all)",
    )
    cancel.add_argument("--yes", action="store_true", help="don't ask first")
    cancel.set_defaults(func=cmd_cancel_jobs)
