"""Stand-in for gcloud, terraform and a text editor, for wildhat's tests.

The test harness puts launchers named gcloud, terraform and fake-editor on
PATH. Each runs this script with the tool's name first. Every call is logged
to calls.jsonl in $WILDHAT_FAKE_DIR, and each tool acts like the real one
closely enough to check wildhat's side of the conversation:

- gcloud keeps a list of signed-in accounts and which one is active, honours
  --account, writes the credential config, and, like google-auth, runs the
  credential config's executable (`wildhat id-token`) whenever a wildhat
  service account's credentials are used. A Google user account's access
  token is "ya29.user:<email>"; its identity token is "id:<email>".
- terraform's plan says there are changes (exit 2) and writes the plan file.
- fake-editor replaces the file with the next text queued in editor.json.

responses.json can override any call: the first entry whose tool matches and
whose words prefix the call's non-flag arguments supplies stdout, stderr and
the exit code instead.
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

FAKE_DIR = Path(os.environ["WILDHAT_FAKE_DIR"])
ID_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"


def load(name: str, default):
    try:
        return json.loads((FAKE_DIR / name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save(name: str, value) -> None:
    (FAKE_DIR / name).write_text(json.dumps(value), encoding="utf-8")


def log(tool: str, args: list[str], **extra) -> None:
    with open(FAKE_DIR / "calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": tool, "args": args, **extra}) + "\n")


def words(args: list[str]) -> list[str]:
    return [a for a in args if not a.startswith("-")]


def option(args: list[str], name: str) -> str | None:
    for arg in args:
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
    return None


def override(tool: str, args: list[str]) -> int | None:
    for response in load("responses.json", []):
        if response["tool"] == tool and words(args)[: len(response["words"])] == response["words"]:
            sys.stdout.write(response.get("stdout", ""))
            sys.stderr.write(response.get("stderr", ""))
            return response.get("returncode", 0)
    return None


# ---------------------------------------------------------------------------
# gcloud
# ---------------------------------------------------------------------------


def credential_source_token(cred_file: str) -> str:
    """Run the executable credential source the way google-auth does."""
    if os.environ.get("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES") != "1":
        raise SystemExit(
            "ERROR: Executables need to be explicitly allowed (set "
            "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run."
        )
    cred = json.loads(Path(cred_file).read_text(encoding="utf-8"))
    executable = cred["credential_source"]["executable"]
    # google-auth splits the command with shlex in POSIX mode.
    argv = shlex.split(executable["command"])
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=executable["timeout_millis"] / 1000
    )
    log("credential-source", argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    try:
        response = json.loads(result.stdout)
    except ValueError:
        raise SystemExit(f"ERROR: The executable's output isn't JSON: {result.stdout!r}")
    if response.get("version") != 1:
        raise SystemExit("ERROR: Executable response is missing version 1.")
    if not response.get("success"):
        raise SystemExit(
            f"ERROR: Executable returned unsuccessful response: code: "
            f"{response.get('code')}, message: {response.get('message')}."
        )
    if response.get("token_type") != ID_TOKEN_TYPE or not response.get("id_token"):
        raise SystemExit(f"ERROR: Executable response has no ID token: {response}")
    if response.get("expiration_time", 0) <= time.time():
        raise SystemExit("ERROR: The token returned by the executable is expired.")
    return response["id_token"]


def gcloud(args: list[str]) -> int:
    # accounts: {email: {"cred_file": path, or None for a Google user account}}
    state = load("gcloud.json", {"config": {}, "accounts": {}, "active": None})
    cmd = words(args)

    def account() -> str:
        name = option(args, "account") or state["active"]
        if name is None:
            raise SystemExit("ERROR: (gcloud) You do not currently have an active account selected.")
        if name not in state["accounts"]:
            raise SystemExit(f"ERROR: (gcloud) There are no credentials for account [{name}].")
        return name

    def use_credentials(name: str) -> str:
        """An access token for the account, the way gcloud gets one."""
        cred_file = state["accounts"][name]["cred_file"]
        if cred_file is None:
            return f"ya29.user:{name}"
        return f"ya29.sa.{credential_source_token(cred_file)[:8]}"

    if cmd[:3] == ["iam", "workload-identity-pools", "create-cred-config"]:
        output = option(args, "output-file")
        sa = option(args, "service-account")
        cred = {
            "type": "external_account",
            "audience": f"//iam.googleapis.com/{cmd[3]}",
            "subject_token_type": ID_TOKEN_TYPE,
            "token_url": "https://sts.googleapis.com/v1/token",
            "service_account_impersonation_url": (
                "https://iamcredentials.googleapis.com/v1/projects/-/"
                f"serviceAccounts/{sa}:generateAccessToken"
            ),
            "credential_source": {
                "executable": {
                    "command": option(args, "executable-command"),
                    "timeout_millis": int(option(args, "executable-timeout-millis")),
                }
            },
        }
        Path(output).write_text(json.dumps(cred, indent=2), encoding="utf-8")
        print(f"Created credential configuration file [{output}].", file=sys.stderr)
        return 0

    if cmd[:2] == ["auth", "login"]:
        cred_file = option(args, "cred-file")
        cred = json.loads(Path(cred_file).read_text(encoding="utf-8"))
        url = cred["service_account_impersonation_url"]
        name = url.split("/serviceAccounts/")[1].split(":")[0]
        state["accounts"][name] = {"cred_file": cred_file}
        if "--no-activate" not in args:
            state["active"] = name
        save("gcloud.json", state)
        print(f"Authenticated with external account credentials for: [{name}].", file=sys.stderr)
        return 0

    if cmd[:2] == ["auth", "list"]:
        assert "--format=json" in args, args
        print(json.dumps(
            [{"account": a, "status": "ACTIVE" if a == state["active"] else ""} for a in state["accounts"]]
        ))
        return 0

    if cmd[:2] == ["config", "set"]:
        state["config"][cmd[2]] = cmd[3]
        save("gcloud.json", state)
        print(f"Updated property [{cmd[2]}].", file=sys.stderr)
        return 0

    if cmd[:2] == ["auth", "revoke"]:
        if state["accounts"].pop(cmd[2], None) is not None and state["active"] == cmd[2]:
            state["active"] = None
        save("gcloud.json", state)
        print(f"Revoked credentials:\n - {cmd[2]}", file=sys.stderr)
        return 0

    if cmd[:2] == ["auth", "print-access-token"]:
        print(use_credentials(account()))
        return 0

    if cmd[:2] == ["auth", "print-identity-token"]:
        name = account()
        if state["accounts"][name]["cred_file"] is not None:
            raise SystemExit("ERROR: (gcloud) the fake gcloud only makes identity tokens for user accounts")
        print(f"id:{name}")
        return 0

    if cmd[:3] == ["run", "jobs", "list"]:
        use_credentials(account())
        print("Listed 0 items.", file=sys.stderr)
        return 0

    # Artifact Registry, as image names: ["<repo>/<name>:<tag>", ...]
    if cmd[:4] == ["artifacts", "docker", "images", "describe"]:
        use_credentials(account())
        if cmd[4] not in state.get("images", []):
            print(f"ERROR: (gcloud.artifacts.docker.images.describe) Image not found: {cmd[4]}", file=sys.stderr)
            return 1
        print(f"image_summary:\n  fully_qualified_digest: {cmd[4].split(':')[0]}@sha256:abc")
        return 0

    if cmd[:2] == ["builds", "submit"]:
        use_credentials(account())
        folder = Path(cmd[2])
        config = option(args, "config")
        substitutions = dict(s.split("=", 1) for s in option(args, "substitutions").split(","))
        if not (folder / "Dockerfile").is_file() or not Path(config).is_file():
            print(f"ERROR: (gcloud.builds.submit) no Dockerfile or {config}", file=sys.stderr)
            return 1
        state.setdefault("images", []).append(substitutions["_IMAGE"])
        save("gcloud.json", state)
        print(f"Created [https://cloudbuild.googleapis.com/v1/builds/b1].\nSUCCESS", file=sys.stderr)
        return 0

    print(f"ERROR: the fake gcloud doesn't know {args}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# terraform and the editor
# ---------------------------------------------------------------------------


def terraform(args: list[str]) -> int:
    command = words(args)[0]
    if command == "plan":
        out = next(a.split("=", 1)[1] for a in args if a.startswith("-out="))
        Path(out).write_text("plan", encoding="utf-8")
        return 2
    if command == "apply":
        if not Path(args[-1]).exists():
            print(f"Error: Failed to load plan file {args[-1]}", file=sys.stderr)
            return 1
    return 0


def editor(args: list[str]) -> int:
    path = Path(args[-1])
    edits = load("editor.json", [])
    if not edits:
        print("fake-editor: no edit queued", file=sys.stderr)
        return 1
    text = edits.pop(0)
    save("editor.json", edits)
    if text is not None:
        path.write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    tool, args = sys.argv[1], sys.argv[2:]
    log(
        tool,
        args,
        allow_executables=os.environ.get("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES"),
        oauth_token=os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN"),
    )
    code = override(tool, args)
    if code is not None:
        return code
    return {"gcloud": gcloud, "terraform": terraform, "fake-editor": editor}[tool](args)


if __name__ == "__main__":
    sys.exit(main())
