"""The wildhat command: sign students in to the team's Google Cloud project.

How sign-in works, in short:

1. The student signs in to Firebase Authentication with an email link sent to
   their school address. Firebase gives us an ID token (good for 1 hour) and a
   refresh token (good until revoked).
2. Google Cloud's workload identity federation trusts Firebase ID tokens. gcloud
   and the Python client libraries run `wildhat id-token` whenever they need a
   fresh one, trade it for a Google access token, and act as the student's own
   service account, wh-<handle>.

Mentors are on the roster too and sign in the same way, so their jobs look
exactly like students'. Their own Google account stays signed in to gcloud
for the admin commands, so every wildhat gcloud call names its account.

Standard library only, and it must work on macOS and Windows.
"""

import argparse
import base64
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from wildhat import __version__

# The project's settings, shared with Terraform (../../../terraform/main.tf).
# None of it is secret, so it is fine in a public repo; ../../../SETUP.md says
# how to fill it in for a new project. The tests point WILDHAT_CONFIG at their
# own copy.
CONFIG = json.loads(
    Path(os.environ.get("WILDHAT_CONFIG") or Path(__file__).with_name("config.json")).read_text(
        encoding="utf-8"
    )
)
PROJECT_ID = CONFIG["project_id"]
PROJECT_NUMBER = CONFIG["project_number"]
REGION = CONFIG["region"]
# Identifies the project to Firebase; it grants no access by itself.
FIREBASE_API_KEY = CONFIG["firebase_api_key"]
POOL = "students"
PROVIDER = "firebase"

PROVIDER_RESOURCE = (
    f"projects/{PROJECT_NUMBER}/locations/global/workloadIdentityPools/"
    f"{POOL}/providers/{PROVIDER}"
)
IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1"
SECURE_TOKEN = "https://securetoken.googleapis.com/v1/token"

# Reuse a cached ID token only if it has at least this many seconds left.
TOKEN_MIN_LIFETIME = 5 * 60
# Handles are first-last: lowercase letters and hyphens, maybe a trailing
# number (jane-doe2), at most 27 characters so wh-<handle> fits in a service
# account ID.
HANDLE_RE = re.compile(r"^[a-z][a-z-]*[0-9]*$")
HANDLE_MAX = 27
ID_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"
# gcloud and google-auth refuse to run an executable credential source unless
# this is set.
ALLOW_EXECUTABLES = "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES"
# Where the Python client libraries look for credentials.
GOOGLE_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

IS_WINDOWS = os.name == "nt"


class WildhatError(Exception):
    """An error with a message meant for the student."""


# ---------------------------------------------------------------------------
# Per-user state
# ---------------------------------------------------------------------------


def state_dir() -> Path:
    if IS_WINDOWS:
        return Path(os.environ["APPDATA"]) / "wildhat"
    return Path.home() / ".config" / "wildhat"


def refresh_token_path() -> Path:
    return state_dir() / "refresh_token"


def id_token_path() -> Path:
    return state_dir() / "id_token.json"


def account_path() -> Path:
    return state_dir() / "account.json"


def cred_path() -> Path:
    return state_dir() / "cred.json"


def ensure_state_dir() -> Path:
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    if not IS_WINDOWS:
        d.chmod(0o700)
    return d


def write_private(path: Path, text: str) -> None:
    """Write a file only the current user can read, replacing it atomically.

    gcloud may run `wildhat id-token` several times at once, so a reader must
    never see a half-written file.
    """
    ensure_state_dir()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if IS_WINDOWS:
        # %APPDATA% is already private to the user.
        tmp.write_text(text, encoding="utf-8")
    else:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_account() -> dict:
    account = read_json(account_path())
    if not account or "handle" not in account:
        raise WildhatError("You're not signed in. Run: wildhat login")
    return account


def service_account(handle: str) -> str:
    return f"wh-{handle}@{PROJECT_ID}.iam.gserviceaccount.com"


# ---------------------------------------------------------------------------
# Firebase Authentication REST calls
# ---------------------------------------------------------------------------


def api_key() -> str:
    if not FIREBASE_API_KEY:
        raise WildhatError(
            "wildhat's config.json has no firebase_api_key. Ask a mentor."
        )
    return FIREBASE_API_KEY


# Firebase error codes, translated for people.
FIREBASE_ERRORS = {
    "INVALID_EMAIL": "That doesn't look like an email address.",
    "INVALID_OOB_CODE": "That sign-in link is invalid or was already used. "
    "Run wildhat login again.",
    "EXPIRED_OOB_CODE": "That sign-in link has expired. Run wildhat login again.",
    "USER_DISABLED": "Your account has been disabled. Ask a mentor.",
    "TOKEN_EXPIRED": "Your sign-in has expired. Run wildhat login again.",
    "INVALID_REFRESH_TOKEN": "Your sign-in is no longer valid. "
    "Run wildhat login again.",
    "USER_NOT_FOUND": "Your account no longer exists. Ask a mentor.",
}


def post(url: str, body: bytes, content_type: str) -> dict:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        with e:  # Close the error's body, or Python warns later.
            try:
                code = json.load(e)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                code = f"HTTP {e.code}"
        # Codes can carry detail after a colon, e.g. "INVALID_EMAIL : ...".
        short = code.split(":")[0].strip()
        raise WildhatError(FIREBASE_ERRORS.get(short, f"Sign-in failed: {code}"))
    except urllib.error.URLError as e:
        raise WildhatError(f"Couldn't reach Google ({e.reason}). Are you online?")


def post_json(url: str, payload: dict) -> dict:
    return post(url, json.dumps(payload).encode(), "application/json")


def send_sign_in_link(email: str, continue_url: str) -> None:
    post_json(
        f"{IDENTITY_TOOLKIT}/accounts:sendOobCode?key={api_key()}",
        {
            "requestType": "EMAIL_SIGNIN",
            "email": email,
            "continueUrl": continue_url,
            "canHandleCodeInApp": True,
        },
    )


def sign_in_with_code(email: str, oob_code: str) -> dict:
    return post_json(
        f"{IDENTITY_TOOLKIT}/accounts:signInWithEmailLink?key={api_key()}",
        {"email": email, "oobCode": oob_code},
    )


def refresh_id_token(refresh_token: str) -> dict:
    body = urllib.parse.urlencode(
        {"grant_type": "refresh_token", "refresh_token": refresh_token}
    ).encode()
    return post(
        f"{SECURE_TOKEN}?key={api_key()}",
        body,
        "application/x-www-form-urlencoded",
    )


def jwt_claims(token: str) -> dict:
    """Decode a JWT's claims without checking its signature.

    We only use the claims to pick a service account name. IAM checks the
    token for real when gcloud exchanges it.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError):
        raise WildhatError("Got a malformed ID token from Firebase.")


def save_id_token(id_token: str, expires_in) -> int:
    expires_at = int(time.time()) + int(expires_in)
    write_private(
        id_token_path(),
        json.dumps({"id_token": id_token, "expires_at": expires_at}),
    )
    return expires_at


# ---------------------------------------------------------------------------
# Getting the code out of the sign-in link
# ---------------------------------------------------------------------------

# Query parameters that mail systems use to wrap the real link, e.g.
# google.com/url?q=... or Outlook safelinks ?url=...
WRAPPER_PARAMS = ("q", "url", "u", "link", "target", "continueUrl")


def find_oob_code(link: str, depth: int = 3) -> str | None:
    """Find the oobCode in a sign-in link, unwrapping up to `depth` redirects."""
    link = link.strip().strip("<>\"'")
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query)
    if params.get("oobCode"):
        return params["oobCode"][0]
    if depth == 0:
        return None
    # parse_qs has already undone one layer of percent-encoding.
    for name in WRAPPER_PARAMS:
        for value in params.get(name, []):
            if "://" in value:
                code = find_oob_code(value, depth - 1)
                if code:
                    return code
    return None


def start_callback_server(codes: queue.Queue) -> HTTPServer:
    """Start a one-shot web server that catches the browser's redirect.

    It listens on 127.0.0.1 only. Listening on all interfaces would make
    Windows ask about the firewall.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            code = find_oob_code(self.path)
            if code:
                self.reply(200, "Signed in to wildhat. You can close this tab.")
                codes.put(code)
            else:
                self.reply(404, "This isn't a wildhat sign-in link.")

        def reply(self, status: int, message: str) -> None:
            body = (
                "<!doctype html><meta charset=utf-8><title>wildhat</title>"
                f"<p style='font: 18px sans-serif; margin: 3em'>{message}</p>"
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # Keep the terminal quiet.

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def read_lines(lines: queue.Queue) -> None:
    """Put each typed line on `lines`, then None at the end of input.

    One thread reads the keyboard for all of login, so a question asked after
    the browser delivers the sign-in link doesn't race a read that's still
    waiting for a pasted one.
    """
    while True:
        try:
            lines.put(input())
        except (EOFError, OSError):
            lines.put(None)
            return


def wait_for_code(codes: queue.Queue, lines: queue.Queue) -> str:
    """Wait for the browser's redirect, or for the student to paste the link
    (for students who can't open the email on this computer)."""
    typing = True
    while True:
        # Wait in short steps: on Windows a bare get() can't be stopped with
        # Ctrl+C.
        try:
            return codes.get(timeout=0.5)
        except queue.Empty:
            pass
        while typing and not lines.empty():
            line = lines.get()
            if line is None:
                lines.put(None)  # Leave the end of input for later questions.
                typing = False
            elif code := find_oob_code(line):
                return code
            elif line.strip():
                print("Couldn't find a sign-in code in that. Paste the whole link.")


def ask_yes(question: str, lines: queue.Queue) -> bool:
    print(f"{question} [Y/n] ", end="", flush=True)
    line = lines.get()
    if line is None:
        lines.put(None)
        print()
        return False
    return line.strip().lower() in ("", "y", "yes")


# ---------------------------------------------------------------------------
# gcloud
# ---------------------------------------------------------------------------


def gcloud_path() -> str:
    # On Windows gcloud is gcloud.cmd, which subprocess can't find by bare
    # name; shutil.which can.
    path = shutil.which("gcloud")
    if not path:
        raise WildhatError(
            "Couldn't find gcloud. Install the Google Cloud CLI: "
            "https://cloud.google.com/sdk/docs/install"
        )
    return path


def gcloud(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, **{ALLOW_EXECUTABLES: "1"})
    result = subprocess.run(
        [gcloud_path(), *args, "--quiet"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise WildhatError(
            f"gcloud {args[0]} {args[1]} failed:\n{result.stderr.strip()}"
        )
    return result


def gcloud_accounts() -> list[dict]:
    """Accounts signed in to gcloud: [{"account": ..., "status": "ACTIVE" or ""}]."""
    return json.loads(gcloud("auth", "list", "--format=json").stdout or "[]")


def is_service_account(account: str) -> bool:
    return account.endswith(".gserviceaccount.com")


def wildhat_command() -> str:
    """The absolute path to this program, for gcloud to run as `id-token`."""
    path = shutil.which("wildhat")
    if not path:
        argv0 = Path(sys.argv[0])
        if argv0.suffix.lower() in ("", ".exe") and argv0.exists():
            path = str(argv0.resolve())
    if not path:
        raise WildhatError(
            "Couldn't find the wildhat program on your PATH. Install it with "
            "`uv tool install ./ai/wildhat` and make sure uv's tool directory "
            "is on your PATH (`uv tool update-shell`)."
        )
    # google-auth splits the command with shlex.split in POSIX mode, which
    # would eat Windows backslashes and break paths with spaces
    # (C:\Users\Jane Doe\...). Single-quoting the path protects both.
    return shlex.quote(str(Path(path).resolve()))


# ---------------------------------------------------------------------------
# Environment variables, so gcloud and the Python libraries work in any terminal
# ---------------------------------------------------------------------------

# Marks wildhat's lines in ~/.zshrc etc., so logout can take them out again.
RC_BEGIN = "# >>> wildhat >>>"
RC_END = "# <<< wildhat <<<"
RC_BLOCK = re.compile(rf"^{re.escape(RC_BEGIN)}\n.*?^{re.escape(RC_END)}\n?", re.M | re.S)


def read_user_env(name: str) -> str | None:
    """A Windows user environment variable, as `setx` saves it."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0]
    except FileNotFoundError:
        return None


def write_user_env(name: str, value: str | None) -> None:
    """Set a Windows user environment variable, or delete it (None), like
    `setx`. Terminals opened afterwards see the change."""
    import ctypes
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        if value is None:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                pass
        else:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    # Tell Windows (HWND_BROADCAST, WM_SETTINGCHANGE), so new terminals get it.
    ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 2, 5000, None)


def shell_rc() -> Path | None:
    """The startup file of the person's shell, if we know it."""
    shell = Path(os.environ.get("SHELL", "")).name
    if shell == "zsh":
        return Path.home() / ".zshrc"
    if shell == "bash":
        # macOS's Terminal starts login shells, which read .bash_profile.
        return Path.home() / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
    return None


def set_up_environment(lines: queue.Queue) -> None:
    """Set the variables gcloud and the Python libraries need in new terminals.

    wildhat itself doesn't need them: it sets ALLOW_EXECUTABLES whenever it
    runs gcloud.
    """
    cred = str(cred_path())
    wanted = {ALLOW_EXECUTABLES: "1", GOOGLE_CREDENTIALS: cred}
    existing = (read_user_env(GOOGLE_CREDENTIALS) if IS_WINDOWS else None) or os.environ.get(
        GOOGLE_CREDENTIALS
    )
    if existing and existing != cred:
        # Someone's own credentials (a mentor's, say). Don't take them over.
        del wanted[GOOGLE_CREDENTIALS]
        print(f"\nLeaving {GOOGLE_CREDENTIALS} set to {existing}.")
        print("The Python libraries will use that instead of wildhat.")

    if IS_WINDOWS:
        for name, value in wanted.items():
            write_user_env(name, value)
        print(f"\nSet {' and '.join(wanted)} for your Windows account.")
        print("Open a new terminal to use gcloud directly.")
        return

    exports = [f"export {name}={shlex.quote(value)}" for name, value in wanted.items()]
    rc = shell_rc()
    text = rc.read_text(encoding="utf-8") if rc and rc.exists() else ""
    if rc is None or RC_BEGIN not in text:
        where = f"~/{rc.name}" if rc else "your shell's startup file"
        print(f"\nTo use gcloud directly in other terminals, {where} needs:\n")
        for line in exports:
            print(f"  {line}")
        print()
        if rc is None or not ask_yes(f"Add them to {where}?", lines):
            return
    text = RC_BLOCK.sub("", text)
    if text and not text.endswith("\n"):
        text += "\n"
    rc.write_text(text + "\n".join([RC_BEGIN, *exports, RC_END]) + "\n", encoding="utf-8")
    print(f"Added them to ~/{rc.name}. Open a new terminal to use gcloud directly.")


def remove_environment() -> None:
    """Undo set_up_environment, as far as it's still wildhat's."""
    if IS_WINDOWS:
        # ALLOW_EXECUTABLES stays: it's harmless, and something else may need it.
        if read_user_env(GOOGLE_CREDENTIALS) == str(cred_path()):
            write_user_env(GOOGLE_CREDENTIALS, None)
        return
    rc = shell_rc()
    if rc and rc.exists():
        text = rc.read_text(encoding="utf-8")
        if RC_BEGIN in text:
            rc.write_text(RC_BLOCK.sub("", text), encoding="utf-8")
            print(f"Removed wildhat's lines from ~/{rc.name}.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_login(args) -> int:
    email = (args.email or input("School email: ")).strip().lower()
    if "@" not in email:
        raise WildhatError("That doesn't look like an email address.")
    gcloud_path()  # Fail now rather than after the email round trip.
    command = wildhat_command()

    codes: queue.Queue = queue.Queue()
    server = start_callback_server(codes)
    port = server.server_address[1]
    try:
        send_sign_in_link(email, f"http://localhost:{port}/")
        print(f"\nSent a sign-in link to {email}.")
        print("Open your school Gmail (gmail.com) in a browser on THIS computer")
        print("and click the link. Sign-in finishes here by itself.")
        print("(If you can only open the email somewhere else, paste the whole link here.)\n")
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=read_lines, args=(lines,), daemon=True).start()
        oob_code = wait_for_code(codes, lines)
    finally:
        server.shutdown()
        server.server_close()

    reply = sign_in_with_code(email, oob_code)
    handle = jwt_claims(reply["idToken"]).get("handle")
    if not handle:
        raise WildhatError(
            "Your email isn't on the team roster yet. Ask a mentor to add you."
        )
    if not HANDLE_RE.match(handle) or len(handle) > HANDLE_MAX:
        raise WildhatError(f"Your roster handle {handle!r} is malformed. Ask a mentor.")

    write_private(refresh_token_path(), reply["refreshToken"])
    save_id_token(reply["idToken"], reply["expiresIn"])
    write_private(
        account_path(),
        json.dumps({"email": email, "handle": handle, "local_id": reply["localId"]}),
    )
    print(f"Signed in as {email} ({handle}). Setting up gcloud...")

    sa = service_account(handle)
    # If a Google account is already active in gcloud (a mentor's, say), keep
    # it active: the admin commands and the person's own gcloud use rely on
    # it. wildhat names the service account whenever it needs it.
    active_user = next(
        (
            a["account"]
            for a in gcloud_accounts()
            if a["status"] == "ACTIVE" and not is_service_account(a["account"])
        ),
        None,
    )
    gcloud(
        "iam", "workload-identity-pools", "create-cred-config", PROVIDER_RESOURCE,
        f"--service-account={sa}",
        f"--executable-command={command} id-token",
        "--executable-timeout-millis=10000",
        f"--output-file={cred_path()}",
    )
    gcloud("auth", "login", f"--cred-file={cred_path()}", *(["--no-activate"] if active_user else []))
    gcloud("config", "set", "project", PROJECT_ID)
    gcloud("config", "set", "run/region", REGION)

    check = gcloud(
        "run", "jobs", "list", f"--project={PROJECT_ID}", f"--region={REGION}", f"--account={sa}", check=False
    )
    if check.returncode != 0:
        raise WildhatError(
            "You're on the roster, but your cloud access isn't set up yet. If you "
            "were added in the last 10 minutes, Google may still be switching it "
            "on: wait a few minutes and run `wildhat login` again. Otherwise, ask "
            "a mentor.\n\ngcloud said:\n" + check.stderr.strip()
        )
    if active_user:
        print(f"Done. wildhat works as {sa}.")
        print(f"gcloud's active account is still {active_user}. To run gcloud as")
        print(f"{sa}, add --account={sa}.")
    else:
        print(f"Done. gcloud now works as {sa}.")
    set_up_environment(lines)
    return 0


def cmd_id_token(args) -> int:
    """Print an ID token in the format gcloud and google-auth expect.

    This is run by gcloud, not by people, so its only output is the JSON.
    """

    def fail(code: str, message: str) -> int:
        print(json.dumps({"version": 1, "success": False, "code": code, "message": message}))
        return 1

    cached = read_json(id_token_path()) or {}
    if cached.get("expires_at", 0) - time.time() > TOKEN_MIN_LIFETIME:
        id_token, expires_at = cached["id_token"], cached["expires_at"]
    else:
        try:
            refresh_token = refresh_token_path().read_text(encoding="utf-8").strip()
        except OSError:
            return fail("not_logged_in", "run wildhat login")
        try:
            reply = refresh_id_token(refresh_token)
        except WildhatError as e:
            return fail("refresh_failed", f"{e} (run wildhat login)")
        if reply.get("refresh_token") and reply["refresh_token"] != refresh_token:
            write_private(refresh_token_path(), reply["refresh_token"])
        id_token = reply["id_token"]
        expires_at = save_id_token(id_token, reply["expires_in"])

    print(
        json.dumps(
            {
                "version": 1,
                "success": True,
                "token_type": ID_TOKEN_TYPE,
                "id_token": id_token,
                "expiration_time": expires_at,
            }
        )
    )
    return 0


def cmd_whoami(args) -> int:
    account = load_account()
    sa = service_account(account["handle"])
    print(f"email:           {account['email']}")
    print(f"handle:          {account['handle']}")
    print(f"service account: {sa}")
    # Getting an access token runs the whole chain: Firebase refresh, token
    # exchange and service account impersonation.
    check = gcloud("auth", "print-access-token", f"--account={sa}", check=False)
    if check.returncode == 0:
        print("access:          working")
        return 0
    print("access:          NOT working")
    print(check.stderr.strip())
    return 1


def cmd_logout(args) -> int:
    account = read_json(account_path())
    if account and "handle" in account and shutil.which("gcloud"):
        gcloud("auth", "revoke", service_account(account["handle"]), check=False)
    if state_dir().exists():
        shutil.rmtree(state_dir())
    remove_environment()
    print("Signed out.")
    return 0


# How `wildhat --help` groups the commands. Each command's help text is the
# one it's added with.
HELP_SECTIONS = [
    ("Sign in", ["login", "whoami", "logout"]),
    ("Run in the cloud", ["launch", "runs", "logs", "cancel", "download"]),
    ("Mentors", ["admin"]),
]


def sectioned_help(sub) -> str:
    """The commands' help, under HELP_SECTIONS' headings. (argparse can only
    list subcommands in one block.)"""
    helps = {action.dest: action.help for action in sub._choices_actions}
    width = max(map(len, helps)) + 2
    lines = ["Team 100 Wildhat AI tools."]
    for title, names in HELP_SECTIONS:
        lines += ["", f"{title}:"]
        lines += [f"  {name.ljust(width)}{helps[name]}" for name in names]
    lines += ["", "Run `wildhat COMMAND --help` for a command's options."]
    return "\n".join(lines)


def build_parser() -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    parser = argparse.ArgumentParser(
        prog="wildhat",
        usage="wildhat COMMAND [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", help=argparse.SUPPRESS)

    login = sub.add_parser("login", help="sign in with your school email")
    login.add_argument("--email", help="your school email address")
    login.set_defaults(func=cmd_login)

    # No help, so it isn't listed: gcloud runs it, people never need to.
    sub.add_parser(
        "id-token", description="Print an ID token. gcloud runs this; you don't need to."
    ).set_defaults(func=cmd_id_token)
    sub.add_parser("whoami", help="show who you're signed in as").set_defaults(
        func=cmd_whoami
    )
    sub.add_parser("logout", help="sign out and delete saved credentials").set_defaults(
        func=cmd_logout
    )
    # Imported here because these import from this module.
    from wildhat import admin, launch, runs

    launch.add_parsers(sub)
    runs.add_parsers(sub)
    admin.add_parsers(sub)
    parser.description = sectioned_help(sub)
    return parser, sub


def main(argv: list[str] | None = None) -> int:
    parser, _ = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WildhatError as e:
        print(f"wildhat: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
