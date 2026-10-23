"""Test harness: runs the real wildhat code against fake Google services.

Tests drive wildhat the way people do, through its command line, and fake only
what's outside it:

- Google's web APIs (Firebase Auth, Cloud Storage, the watchdog function),
  by replacing urllib.request.urlopen with FakeGoogle.
- gcloud, terraform and the editor, with fake programs on PATH (fake_tool.py).
  When the fake gcloud uses a student's credentials it runs `wildhat id-token`
  from the credential config, as a separate process, the way google-auth does.
- The keyboard (input()).

Each test gets its own home directory, so it never touches the real wildhat or
gcloud state. The temp paths contain spaces on purpose: real Windows paths
often do.
"""

import base64
import contextlib
import http.client
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"
# Always test this source tree, even if some other wildhat is installed.
sys.path.insert(0, str(SRC))

from wildhat import cli  # noqa: E402

PROJECT = "vernal-mantra-512116-d0"
ADMIN_BUCKET = f"{PROJECT}-admin"
WATCHDOG_URL = "https://job-watchdog-abc123-uc.a.run.app"
MENTOR_ACCESS_TOKEN = "ya29.mentor"
MENTOR_ID_TOKEN = "mentor-id-token"
REAL_URLOPEN = urllib.request.urlopen
# A wildhat run that takes longer than this is stuck (e.g. login waiting for a
# sign-in link that never comes).
RUN_TIMEOUT = 60


def jwt(claims: dict) -> str:
    def part(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256', 'typ': 'JWT'})}.{part(claims)}.signature"


def jwt_claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class FakeGoogle:
    """Google's web APIs, in memory. Unknown requests fail the test."""

    def __init__(self):
        self.users: dict[str, dict] = {}  # Firebase users by localId
        self.objects: dict[tuple[str, str], tuple[bytes, int]] = {}  # GCS
        self.requests: list[dict] = []
        self.sent_links: list[str] = []
        self.browser_visits: list[tuple[str, int, str]] = []
        # Whether the student clicks the emailed link on this computer.
        self.click_links = True
        self.offline = False
        self.failures: dict[str, tuple[int, str]] = {}
        self.watchdog_reply = {"month_cost": 12.5, "checked": 0, "cancelled": []}
        self._oob_codes: dict[str, str] = {}
        self._refresh_tokens: dict[str, str] = {}
        self._generation = 1_700_000_000_000_000
        self._next = 0

    # --- Test setup and inspection ----------------------------------------

    def add_user(self, email: str, handle: str | None = None, **fields) -> str:
        local_id = self._id("user")
        user = {"localId": local_id, "email": email, **fields}
        if handle:
            user["customAttributes"] = json.dumps({"handle": handle})
        self.users[local_id] = user
        return local_id

    def user(self, email: str) -> dict:
        return next(u for u in self.users.values() if u["email"] == email)

    def claims(self, email: str) -> dict:
        return json.loads(self.user(email).get("customAttributes") or "{}")

    def put_object(self, bucket: str, name: str, data: bytes) -> int:
        self._generation += 1
        self.objects[(bucket, name)] = (data, self._generation)
        return self._generation

    def object(self, bucket: str, name: str) -> bytes | None:
        found = self.objects.get((bucket, name))
        return found[0] if found else None

    def fail(self, path_suffix: str, status: int, message: str) -> None:
        """Make requests whose path ends with `path_suffix` fail."""
        self.failures[path_suffix] = (status, message)

    def requests_to(self, path_suffix: str) -> list[dict]:
        return [r for r in self.requests if r["path"].endswith(path_suffix)]

    # --- urlopen ------------------------------------------------------------

    def urlopen(self, req, timeout=None, **kwargs):
        if isinstance(req, str):
            req = urllib.request.Request(req)
        url = urllib.parse.urlsplit(req.full_url)
        if url.hostname in ("localhost", "127.0.0.1"):
            return REAL_URLOPEN(req, timeout=timeout, **kwargs)
        if self.offline:
            raise urllib.error.URLError("[Errno 8] nodename nor servname provided")
        request = {
            "method": req.get_method(),
            "host": url.hostname,
            "path": urllib.parse.unquote(url.path),
            "query": dict(urllib.parse.parse_qsl(url.query)),
            "headers": {k.lower(): v for k, v in req.header_items()},
            "body": req.data,
        }
        self.requests.append(request)
        for suffix, (status, message) in self.failures.items():
            if request["path"].endswith(suffix):
                return self._error(req.full_url, status, message)
        handler = {
            "identitytoolkit.googleapis.com": self._identity_toolkit,
            "securetoken.googleapis.com": self._secure_token,
            "storage.googleapis.com": self._storage,
            "cloudfunctions.googleapis.com": self._cloud_functions,
            urllib.parse.urlsplit(WATCHDOG_URL).hostname: self._watchdog,
        }.get(url.hostname)
        if handler is None:
            raise AssertionError(f"Unexpected request: {request['method']} {req.full_url}")
        status, body, headers = handler(request)
        if status >= 400:
            return self._error(req.full_url, status, body)
        return self._response(body, headers)

    def _response(self, body, headers=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        message = http.client.HTTPMessage()
        for name, value in (headers or {}).items():
            message[name] = value
        response = io.BytesIO(body)
        response.headers = message
        response.status = 200
        return response

    def _error(self, url: str, status: int, message: str):
        body = json.dumps({"error": {"code": status, "message": message}}).encode()
        raise urllib.error.HTTPError(url, status, message, http.client.HTTPMessage(), io.BytesIO(body))

    def _id(self, kind: str) -> str:
        self._next += 1
        return f"{kind}-{self._next}"

    def _id_token(self, user: dict) -> str:
        claims = {
            "iss": f"https://securetoken.google.com/{PROJECT}",
            "aud": PROJECT,
            "sub": user["localId"],
            "email": user["email"],
            "exp": int(time.time()) + 3600,
            "n": self._id("token"),
        }
        claims.update(json.loads(user.get("customAttributes") or "{}"))
        return jwt(claims)

    def _is_mentor(self, request: dict) -> bool:
        return (
            request["headers"].get("authorization") == f"Bearer {MENTOR_ACCESS_TOKEN}"
            and request["headers"].get("x-goog-user-project") == PROJECT
        )

    # --- Firebase Authentication ------------------------------------------

    def _identity_toolkit(self, request: dict):
        path = request["path"]
        if path.startswith("/v1/accounts:"):
            if request["query"].get("key") != cli.FIREBASE_API_KEY:
                return 400, "API key not valid. Please pass a valid API key.", None
            body = json.loads(request["body"])
            if path == "/v1/accounts:sendOobCode":
                return self._send_oob_code(body)
            if path == "/v1/accounts:signInWithEmailLink":
                return self._sign_in_with_email_link(body)
        prefix = f"/v1/projects/{PROJECT}/accounts"
        if path.startswith(prefix):
            if not self._is_mentor(request):
                return 403, "The caller does not have permission", None
            body = json.loads(request["body"]) if request["body"] else {}
            if path == prefix + ":batchGet":
                return 200, {"kind": "identitytoolkit#DownloadAccountResponse", "users": list(self.users.values())}, None
            if path == prefix and request["method"] == "POST":
                local_id = self._id("user")
                self.users[local_id] = {"localId": local_id, **body}
                return 200, {"localId": local_id, "email": body["email"]}, None
            if path == prefix + ":update":
                user = self.users[body["localId"]]
                if "displayName" in body:
                    user["displayName"] = body["displayName"]
                if "customAttributes" in body:
                    user["customAttributes"] = body["customAttributes"]
                if "disableUser" in body:
                    user["disabled"] = body["disableUser"]
                return 200, {"localId": user["localId"]}, None
        raise AssertionError(f"Unexpected Identity Toolkit call: {request['method']} {path}")

    def _send_oob_code(self, body: dict):
        assert body["requestType"] == "EMAIL_SIGNIN" and body["canHandleCodeInApp"]
        code = self._id("oob")
        self._oob_codes[code] = body["email"]
        params = {"apiKey": cli.FIREBASE_API_KEY, "oobCode": code, "mode": "signIn", "lang": "en"}
        link = (
            f"https://{PROJECT}.firebaseapp.com/__/auth/action?"
            + urllib.parse.urlencode({**params, "continueUrl": body["continueUrl"]})
        )
        self.sent_links.append(link)
        if self.click_links:
            # Firebase's page sends the browser on to continueUrl with the code.
            target = body["continueUrl"] + "?" + urllib.parse.urlencode(params)
            threading.Thread(target=self._browse, args=(target,), daemon=True).start()
        return 200, {"kind": "identitytoolkit#GetOobConfirmationCodeResponse", "email": body["email"]}, None

    def _browse(self, target: str) -> None:
        # Browsers ask for a favicon too.
        root = urllib.parse.urlsplit(target)._replace(path="/favicon.ico", query="").geturl()
        for url in (root, target):
            try:
                with REAL_URLOPEN(url, timeout=10) as resp:
                    self.browser_visits.append((url, resp.status, resp.read().decode()))
            except urllib.error.HTTPError as e:
                self.browser_visits.append((url, e.code, e.read().decode()))

    def _sign_in_with_email_link(self, body: dict):
        if self._oob_codes.pop(body["oobCode"], None) != body["email"]:
            return 400, "INVALID_OOB_CODE", None
        try:
            user = self.user(body["email"])
        except StopIteration:  # Firebase creates the user on first sign-in.
            local_id = self.add_user(body["email"])
            user = self.users[local_id]
        if user.get("disabled"):
            return 400, "USER_DISABLED", None
        refresh = self._id("refresh")
        self._refresh_tokens[refresh] = user["localId"]
        return 200, {
            "idToken": self._id_token(user),
            "refreshToken": refresh,
            "expiresIn": "3600",
            "localId": user["localId"],
            "email": user["email"],
        }, None

    def _secure_token(self, request: dict):
        assert request["query"].get("key") == cli.FIREBASE_API_KEY
        form = dict(urllib.parse.parse_qsl(request["body"].decode()))
        local_id = self._refresh_tokens.get(form.get("refresh_token"))
        if local_id is None or local_id not in self.users:
            return 400, "INVALID_REFRESH_TOKEN", None
        user = self.users[local_id]
        if user.get("disabled"):
            return 400, "USER_DISABLED", None
        return 200, {
            "id_token": self._id_token(user),
            "refresh_token": form["refresh_token"],
            "expires_in": "3600",
            "user_id": local_id,
        }, None

    # --- Cloud Storage ------------------------------------------------------

    def _storage(self, request: dict):
        if not self._is_mentor(request):
            return 403, "Anonymous caller does not have storage.objects.get access", None
        path = request["path"]
        if path.startswith("/storage/v1/b/") and request["query"].get("alt") == "media":
            bucket, name = path[len("/storage/v1/b/"):].split("/o/", 1)
            found = self.objects.get((bucket, name))
            if not found:
                return 404, "No such object", None
            return 200, found[0], {"X-Goog-Generation": str(found[1])}
        if path.startswith("/upload/storage/v1/b/") and request["method"] == "POST":
            bucket = path[len("/upload/storage/v1/b/"):].split("/o", 1)[0]
            name = request["query"]["name"]
            want = int(request["query"]["ifGenerationMatch"])
            have = self.objects.get((bucket, name), (None, 0))[1]
            if want != have:
                return 412, "At least one of the pre-conditions you specified did not hold.", None
            generation = self.put_object(bucket, name, request["body"])
            return 200, {"name": name, "generation": str(generation)}, None
        raise AssertionError(f"Unexpected Cloud Storage call: {request['method']} {path}")

    # --- The job watchdog ---------------------------------------------------

    def _cloud_functions(self, request: dict):
        if not self._is_mentor(request):
            return 403, "Permission denied", None
        if request["path"] == f"/v2/projects/{PROJECT}/locations/us-central1/functions/job-watchdog":
            return 200, {"serviceConfig": {"uri": WATCHDOG_URL}}, None
        raise AssertionError(f"Unexpected Cloud Functions call: {request['path']}")

    def _watchdog(self, request: dict):
        if request["headers"].get("authorization") != f"Bearer {MENTOR_ID_TOKEN}":
            return 403, "Forbidden", None
        assert request["method"] == "POST"
        return 200, self.watchdog_reply, None


class Run:
    def __init__(self, code: int, out: str, err: str):
        self.code, self.out, self.err = code, out, err

    def __repr__(self):
        return f"Run(code={self.code}, out={self.out!r}, err={self.err!r})"


class WildhatTestCase(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="wildhat test "))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.tmp = tmp
        self.home = tmp / "home dir"
        self.home.mkdir()
        self.bin = tmp / "Jane Doe" / "bin"
        self.bin.mkdir(parents=True)
        self.fake_dir = tmp / "fake"
        self.fake_dir.mkdir()
        self.terraform_dir = tmp / "terraform"
        self.terraform_dir.mkdir()

        for tool in ("gcloud", "terraform", "fake-editor"):
            self.write_launcher(tool, [sys.executable, str(TESTS / "fake_tool.py"), tool])
        # `wildhat` as gcloud runs it. It can't reach FakeGoogle, so send
        # anything it tries online to a closed port rather than to Google.
        self.write_launcher(
            "wildhat",
            [sys.executable, "-m", "wildhat"],
            {"PYTHONPATH": str(SRC), "HTTPS_PROXY": "http://127.0.0.1:9"},
        )

        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                cli.ALLOW_EXECUTABLES,
                "VISUAL",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "WILDHAT_FIREBASE_API_KEY",
            )
        }
        # Keep any real gcloud, terraform or wildhat out of reach (GitHub's
        # Ubuntu runners have gcloud), so only the fakes can ever run.
        path = [
            d
            for d in os.environ["PATH"].split(os.pathsep)
            if d and not any(shutil.which(tool, path=d) for tool in ("gcloud", "terraform", "wildhat"))
        ]
        env.update(
            PATH=os.pathsep.join([str(self.bin), *path]),
            HOME=str(self.home),
            USERPROFILE=str(self.home),
            APPDATA=str(self.home / "AppData" / "Roaming"),
            EDITOR="fake-editor",
            WILDHAT_FAKE_DIR=str(self.fake_dir),
            WILDHAT_TERRAFORM_DIR=str(self.terraform_dir),
        )
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.google = FakeGoogle()
        patcher = mock.patch("urllib.request.urlopen", self.google.urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.typed: list = []
        patcher = mock.patch("builtins.input", self._input)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_launcher(self, name: str, argv: list[str], env: dict | None = None) -> None:
        if os.name == "nt":
            lines = ["@echo off"]
            lines += [f'set "{k}={v}"' for k, v in (env or {}).items()]
            lines += [" ".join(f'"{a}"' for a in argv) + " %*", "exit /b %ERRORLEVEL%"]
            (self.bin / f"{name}.cmd").write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        else:
            lines = ["#!/bin/sh"]
            lines += [f"export {k}='{v}'" for k, v in (env or {}).items()]
            lines += ["exec " + " ".join(f"'{a}'" for a in argv) + ' "$@"']
            path = self.bin / name
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            path.chmod(0o755)

    def remove_launcher(self, name: str) -> None:
        for path in self.bin.glob(f"{name}*"):
            path.unlink()

    def _input(self, prompt: str = "") -> str:
        sys.stdout.write(prompt)
        if not self.typed:
            raise EOFError
        line = self.typed.pop(0)
        return line() if callable(line) else line

    # --- Running wildhat ----------------------------------------------------

    def wildhat(self, *argv: str, typed=()) -> Run:
        """Run `wildhat <argv>`, typing `typed` at its prompts.

        An entry in `typed` can be a function, called at the moment wildhat
        asks, for input that depends on what has happened so far.
        """
        self.typed = list(typed)
        out, err = io.StringIO(), io.StringIO()
        result = {}

        def run():
            try:
                result["code"] = cli.main(list(argv))
            except SystemExit as e:  # argparse errors and --version
                result["code"] = e.code
            except BaseException as e:
                result["error"] = e

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(RUN_TIMEOUT)
        if thread.is_alive():
            self.fail(f"wildhat {' '.join(argv)} hung.\nOutput so far:\n{out.getvalue()}{err.getvalue()}")
        if "error" in result:
            raise result["error"]
        return Run(result["code"], out.getvalue(), err.getvalue())

    def assertSucceeded(self, run: Run) -> None:
        self.assertEqual(run.code, 0, run)

    def assertFailedWith(self, run: Run, message: str) -> None:
        self.assertNotEqual(run.code, 0, run)
        self.assertIn(message, run.err, run)

    # --- Fake programs ------------------------------------------------------

    def calls(self, tool: str) -> list[dict]:
        try:
            lines = (self.fake_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [c for c in map(json.loads, lines) if c["tool"] == tool]

    def commands(self, tool: str) -> list[list[str]]:
        """The non-flag words of each call to `tool`, e.g. ["config", "set", ...]."""
        return [[a for a in c["args"] if not a.startswith("-")] for c in self.calls(tool)]

    def respond(self, tool: str, words: list[str], *, stdout="", stderr="", returncode=0) -> None:
        """Make the fake `tool` answer calls starting with `words` this way."""
        path = self.fake_dir / "responses.json"
        responses = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        responses.append(
            {"tool": tool, "words": words, "stdout": stdout, "stderr": stderr, "returncode": returncode}
        )
        path.write_text(json.dumps(responses), encoding="utf-8")

    def queue_edits(self, *texts: str | None) -> None:
        """What the editor does each time it opens: replace the file with the
        text, or leave it alone (None)."""
        (self.fake_dir / "editor.json").write_text(json.dumps(list(texts)), encoding="utf-8")

    # --- Students -----------------------------------------------------------

    @property
    def state_dir(self) -> Path:
        if os.name == "nt":
            return self.home / "AppData" / "Roaming" / "wildhat"
        return self.home / ".config" / "wildhat"

    def log_in(self, email: str = "813579@seq.org", handle: str = "kai-morales") -> Run:
        self.google.add_user(email, handle)
        run = self.wildhat("login", "--email", email)
        self.assertSucceeded(run)
        return run
