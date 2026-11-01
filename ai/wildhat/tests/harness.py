"""Test harness: runs the real wildhat code against fake Google services.

Tests drive wildhat the way people do, through its command line, and fake only
what's outside it:

- Google's web APIs (Firebase Auth, Cloud Storage, the watchdog function),
  by replacing urllib.request.urlopen with FakeGoogle.
- gcloud, terraform and the editor, with fake programs on PATH (fake_tool.py).
  When the fake gcloud uses a student's credentials it runs `wildhat id-token`
  from the credential config, as a separate process, the way google-auth does.
- The keyboard (input()).
- Windows' user environment variables (the registry functions in cli). The
  real ones are tested on their own, in test_environment.py.

Each test gets its own home directory, so it never touches the real wildhat or
gcloud state. The temp paths contain spaces on purpose: real Windows paths
often do.
"""

import base64
import contextlib
import datetime
import gc
import http.client
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"
# Always test this source tree, even if some other wildhat is installed.
sys.path.insert(0, str(SRC))
# A made-up project, so the tests check that nothing about the real one is
# hard-coded. Every wildhat process the tests start reads it too.
os.environ["WILDHAT_CONFIG"] = str(TESTS / "config.json")

from wildhat import cli  # noqa: E402

PROJECT = cli.PROJECT_ID
REGION = cli.REGION
ADMIN_BUCKET = f"{PROJECT}-admin"
WILDHAT_BUCKET = f"{PROJECT}-wildhat"
WATCHDOG_URL = "https://job-watchdog-abc123-uc.a.run.app"
# The mentor's Google account, as the fake gcloud hands out its tokens.
MENTOR = "dana@team100.org"
MENTOR_ACCESS_TOKEN = f"ya29.user:{MENTOR}"
MENTOR_ID_TOKEN = f"id:{MENTOR}"
REAL_URLOPEN = urllib.request.urlopen
REAL_GIT = shutil.which("git")
# A wildhat run that takes longer than this is stuck (e.g. login waiting for a
# sign-in link that never comes).
RUN_TIMEOUT = 60


def jwt(claims: dict) -> str:
    def part(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256', 'typ': 'JWT'})}.{part(claims)}.signature"


def firestore_fields(fields: dict) -> dict:
    """Firestore's typed values ({"integerValue": "3"}) as plain Python."""

    def plain(value: dict):
        (kind, v), = value.items()
        if kind == "integerValue":
            return int(v)
        if kind == "arrayValue":
            return [plain(x) for x in v.get("values", [])]
        if kind == "mapValue":
            return firestore_fields(v.get("fields", {}))
        return v

    return {k: plain(v) for k, v in fields.items()}


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
        # Firestore documents by path under documents/, as {field: value},
        # and as Firestore's typed values.
        self.documents: dict[str, dict] = {}
        self.typed_documents: dict[str, dict] = {}
        # Cloud Run executions' progress by name, and those cancelled.
        self.execution_progress: dict[str, dict] = {}
        self.cancelled_executions: list[str] = []
        self.log_entries: list[dict] = []
        # Called whenever logs are read, so a test can make a run progress.
        self.on_log_read = None
        # Pages are small, so that paging gets tested.
        self.page_size = 2
        # Google accounts with access to the admin APIs (project Owners).
        self.mentors = {MENTOR}
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
            "url": req.full_url,
        }
        self.requests.append(request)
        for suffix, (status, message) in self.failures.items():
            if request["path"].endswith(suffix):
                return self._error(req.full_url, status, message)
        handler = {
            "identitytoolkit.googleapis.com": self._identity_toolkit,
            "securetoken.googleapis.com": self._secure_token,
            "storage.googleapis.com": self._storage,
            "firestore.googleapis.com": self._firestore,
            "run.googleapis.com": self._run_api,
            "logging.googleapis.com": self._logging,
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
            request["headers"].get("authorization") in {f"Bearer ya29.user:{m}" for m in self.mentors}
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
                with e:
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

    def _service_account(self, request: dict) -> str | None:
        """The handle whose wh- service account made the request, if one did."""
        token = request["headers"].get("authorization", "")
        prefix, suffix = "Bearer ya29.sa:wh-", f"@{PROJECT}.iam.gserviceaccount.com"
        if token.startswith(prefix) and token.endswith(suffix):
            return token[len(prefix):-len(suffix)]
        return None

    def _may_access(self, request: dict, bucket: str, name: str, write: bool) -> bool:
        """IAM for the buckets (ai/terraform/students.tf): mentors may do
        anything; wh- accounts may read the wildhat bucket and write only
        their own src/ and runs/ folders."""
        if self._is_mentor(request):
            return True
        handle = self._service_account(request)
        if handle is None or bucket != WILDHAT_BUCKET:
            return False
        return not write or name.startswith((f"src/{handle}/", f"runs/{handle}/"))

    def _storage(self, request: dict):
        path = request["path"]
        if path.startswith("/storage/v1/b/") and path.endswith("/o") and request["method"] == "GET":
            # Listing, a page of `page_size` at a time so paging gets tested.
            bucket = path[len("/storage/v1/b/"):-len("/o")]
            prefix = request["query"].get("prefix", "")
            if not self._may_access(request, bucket, prefix, write=False):
                return 403, "Caller does not have storage.objects.list access", None
            names = sorted(n for (b, n) in self.objects if b == bucket and n.startswith(prefix))
            start = int(request["query"].get("pageToken") or 0)
            page = names[start:start + self.page_size]
            reply = {"items": [{"name": n, "size": str(len(self.objects[bucket, n][0]))} for n in page]}
            if start + self.page_size < len(names):
                reply["nextPageToken"] = str(start + self.page_size)
            return 200, reply, None
        if path.startswith("/storage/v1/b/") and request["method"] == "GET":
            bucket, name = path[len("/storage/v1/b/"):].split("/o/", 1)
            if not self._may_access(request, bucket, name, write=False):
                return 403, "Caller does not have storage.objects.get access", None
            found = self.objects.get((bucket, name))
            if not found:
                return 404, "No such object", None
            if request["query"].get("alt") != "media":
                return 200, {"name": name, "generation": str(found[1]), "size": str(len(found[0]))}, None
            return 200, found[0], {"X-Goog-Generation": str(found[1])}
        if path.startswith("/upload/storage/v1/b/") and request["method"] == "POST":
            bucket = path[len("/upload/storage/v1/b/"):].split("/o", 1)[0]
            name = request["query"]["name"]
            if not self._may_access(request, bucket, name, write=True):
                return 403, "Caller does not have storage.objects.create access", None
            want = int(request["query"]["ifGenerationMatch"])
            have = self.objects.get((bucket, name), (None, 0))[1]
            if want != have:
                return 412, "At least one of the pre-conditions you specified did not hold.", None
            generation = self.put_object(bucket, name, request["body"])
            return 200, {"name": name, "generation": str(generation)}, None
        raise AssertionError(f"Unexpected Cloud Storage call: {request['method']} {path}")

    # --- Firestore ------------------------------------------------------------

    def update_document(self, path: str, typed: dict) -> None:
        """Set fields of a Firestore document, given as typed values
        ({"stringValue": ...}): what the base image's entrypoint does as an
        episode goes."""
        self.typed_documents.setdefault(path, {}).update(typed)
        self.documents[path] = firestore_fields(self.typed_documents[path])

    def _firestore(self, request: dict):
        # Every wh- account has roles/datastore.user (ai/terraform/students.tf).
        if not self._service_account(request) and not self._is_mentor(request):
            return 403, "Missing or insufficient permissions.", None
        root = f"projects/{PROJECT}/databases/(default)/documents"
        prefix = f"/v1/{root}"
        path = request["path"]
        assert path.startswith(prefix), path
        doc = path[len(prefix):].lstrip("/")

        def as_document(name: str) -> dict:
            return {"name": f"{root}/{name}", "fields": self.typed_documents[name]}

        if path.endswith(":commit"):
            writes = json.loads(request["body"])["writes"]
            names = [w["update"]["name"].split("/documents/", 1)[1] for w in writes]
            for write, name in zip(writes, names):
                if write.get("currentDocument", {}).get("exists") is False and name in self.documents:
                    return 409, f"Document already exists: {name}", None
            for write, name in zip(writes, names):
                self.typed_documents[name] = {}
                self.update_document(name, write["update"]["fields"])
            return 200, {"writeResults": [{} for _ in writes]}, None
        if path.endswith(":runQuery"):
            query = json.loads(request["body"])["structuredQuery"]
            (collection,) = [c["collectionId"] for c in query["from"]]
            assert collection == "runs", query
            assert query["orderBy"] == [{"field": {"fieldPath": "created_at"}, "direction": "DESCENDING"}], query
            names = [n for n in self.typed_documents if n.count("/") == 1 and n.startswith("runs/")
                     and "created_at" in self.typed_documents[n]]
            if "where" in query:
                f = query["where"]["fieldFilter"]
                assert f["field"]["fieldPath"] == "launched_by" and f["op"] == "EQUAL", f
                names = [n for n in names if self.documents[n]["launched_by"] == f["value"]["stringValue"]]
            names.sort(key=lambda n: self.documents[n]["created_at"], reverse=True)
            rows = [{"document": as_document(n)} for n in names[:query["limit"]]]
            return 200, rows or [{"readTime": "2026-10-26T00:00:00Z"}], None
        if request["method"] == "GET" and doc.count("/") % 2 == 0:  # A collection
            names = sorted(n for n in self.typed_documents if n.rsplit("/", 1)[0] == doc)
            start = int(request["query"].get("pageToken") or 0)
            page = names[start:start + self.page_size]
            reply = {"documents": [as_document(n) for n in page]}
            if start + self.page_size < len(names):
                reply["nextPageToken"] = str(start + self.page_size)
            return 200, reply, None
        if request["method"] == "GET":
            if doc not in self.typed_documents:
                return 404, f"Document {doc} not found", None
            return 200, as_document(doc), None
        if request["method"] == "PATCH":
            typed = json.loads(request["body"])["fields"]
            mask = urllib.parse.parse_qs(urllib.parse.urlsplit(request["url"]).query)["updateMask.fieldPaths"]
            assert sorted(mask) == sorted(typed), (mask, typed)
            self.update_document(doc, typed)
            return 200, {"name": doc}, None
        if request["method"] == "DELETE":
            self.documents.pop(doc, None)
            self.typed_documents.pop(doc, None)
            return 200, {}, None
        raise AssertionError(f"Unexpected Firestore call: {request['method']} {path}")

    # --- Cloud Run (the REST API; gcloud's side is in fake_tool.py) ----------

    def executions(self) -> dict[str, dict]:
        """Executions the fake gcloud started, by name, with their progress
        (set_execution) merged in."""
        path = Path(os.environ["WILDHAT_FAKE_DIR"]) / "gcloud.json"
        started = json.loads(path.read_text(encoding="utf-8")).get("executions", []) if path.exists() else []
        found = {}
        for e in started:
            name = e["metadata"]["name"]
            job = name.rsplit("-", 1)[0]
            found[name] = {
                "name": f"projects/{PROJECT}/locations/{REGION}/jobs/{job}/executions/{name}",
                "taskCount": e["spec"]["taskCount"],
                "createTime": "2026-10-26T07:00:00Z",
                **self.execution_progress.get(name, {}),
            }
        return found

    def set_execution(self, name: str, **progress) -> None:
        """E.g. set_execution(name, runningCount=2, succeededCount=1), or
        completionTime=... once it's over. Counts that are 0 are left out, as
        Cloud Run does."""
        self.execution_progress.setdefault(name, {}).update(progress)

    def _run_api(self, request: dict):
        # run.viewer on the project lets every wh- account read executions.
        handle = self._service_account(request)
        if handle is None:
            return 403, "Permission denied", None
        m = re.fullmatch(rf"/v2/projects/{PROJECT}/locations/{REGION}/jobs/([^/]+)/executions/([^/:]+)(:cancel)?",
                         request["path"])
        assert m, request["path"]
        job, name, cancel = m.groups()
        execution = self.executions().get(name)
        if execution is None or not name.startswith(job + "-"):
            return 404, f"Execution {name} not found", None
        if not cancel:
            assert request["method"] == "GET"
            return 200, execution, None
        # run.jobsExecutorWithOverrides on their own jobs only (jobs.tf).
        if job.removeprefix("wh-").removesuffix("-gpu") != handle:
            return 403, "Permission 'run.executions.cancel' denied", None
        done = sum(execution.get(f"{k}Count", 0) for k in ("succeeded", "failed"))
        self.set_execution(name, completionTime="2026-10-26T08:00:00Z", runningCount=0,
                           cancelledCount=execution["taskCount"] - done)
        self.cancelled_executions.append(name)
        return 200, {"name": execution["name"]}, None

    # --- Cloud Logging --------------------------------------------------------

    def add_log(self, execution: str, episode: int | None, text: str, log="run.googleapis.com%2Fstdout",
                second: int | None = None) -> None:
        """A line one episode printed, as Cloud Run records it, `second`
        seconds into the run (by default a second after the last line).
        Cloud Run also writes audit entries for an execution, with no
        episode."""
        if not self.log_entries:
            self._log_start = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        if second is None:
            second = len(self.log_entries)
        when = self._log_start + datetime.timedelta(seconds=second)
        self.log_entries.append({
            "insertId": f"log-{len(self.log_entries)}",
            "timestamp": when.isoformat().replace("+00:00", "Z"),
            "logName": f"projects/{PROJECT}/logs/{log}",
            **({"textPayload": text} if text is not None else {"protoPayload": {"@type": "AuditLog"}}),
            "labels": {"run.googleapis.com/execution_name": execution,
                       **({} if episode is None else {"run.googleapis.com/task_index": str(episode)})},
        })

    def _logging(self, request: dict):
        if self._service_account(request) is None:  # logging.viewer
            return 403, "Permission denied", None
        assert request["path"] == "/v2/entries:list", request["path"]
        if self.on_log_read:
            self.on_log_read()
        body = json.loads(request["body"])
        assert body["resourceNames"] == [f"projects/{PROJECT}"] and body["orderBy"] == "timestamp asc", body
        flt = body["filter"]
        assert 'resource.type="cloud_run_job"' in flt, flt
        execution = re.search(r'labels."run.googleapis.com/execution_name"="([^"]+)"', flt).group(1)
        task = re.search(r'labels."run.googleapis.com/task_index"="(\d+)"', flt)
        log = re.search(r'logName:"([^"]+)"', flt)
        after = re.search(r'timestamp>="([^"]+)"', flt).group(1)
        entries = sorted((
            e for e in self.log_entries
            if e["labels"]["run.googleapis.com/execution_name"] == execution
            and (task is None or e["labels"].get("run.googleapis.com/task_index") == task.group(1))
            and (log is None or log.group(1) in e["logName"])
            and e["timestamp"] >= after
        ), key=lambda e: e["timestamp"])
        start = int(body.get("pageToken") or 0)
        reply = {"entries": entries[start:start + self.page_size]}
        if start + self.page_size < len(entries):
            reply["nextPageToken"] = str(start + self.page_size)
        return 200, reply, None

    # --- The job watchdog ---------------------------------------------------

    def _cloud_functions(self, request: dict):
        if not self._is_mentor(request):
            return 403, "Permission denied", None
        if request["path"] == f"/v2/projects/{PROJECT}/locations/{REGION}/functions/job-watchdog":
            return 200, {"serviceConfig": {"uri": WATCHDOG_URL}}, None
        raise AssertionError(f"Unexpected Cloud Functions call: {request['path']}")

    def _watchdog(self, request: dict):
        if request["headers"].get("authorization") not in {f"Bearer id:{m}" for m in self.mentors}:
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
        # The real git, for launch. Its folder may be hidden below for also
        # holding a real gcloud (/usr/bin on GitHub's Ubuntu runners).
        if REAL_GIT:
            self.write_launcher("git", [REAL_GIT])
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
            SHELL="/bin/zsh",
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
        self.output = io.StringIO()
        patcher = mock.patch("builtins.input", self._input)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Windows user environment variables (what `setx` sets), so tests
        # never change the real ones.
        self.user_env: dict[str, str] = {}
        for name, fake in [
            ("read_user_env", self.user_env.get),
            ("write_user_env", self._write_user_env),
        ]:
            patcher = mock.patch.object(cli, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_user_env(self, name: str, value: str | None) -> None:
        if value is None:
            self.user_env.pop(name, None)
        else:
            self.user_env[name] = value

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

    def when_asked(self, question: str, answer: str):
        """For `typed`: wait until wildhat has asked `question`, then answer.
        wildhat reads the keyboard ahead, so a plain string would be typed
        before the question appears."""

        def type_it() -> str:
            deadline = time.monotonic() + RUN_TIMEOUT
            while question not in self.output.getvalue():
                if time.monotonic() > deadline:
                    raise EOFError
                time.sleep(0.01)
            return answer

        return type_it

    # --- Running wildhat ----------------------------------------------------

    def wildhat(self, *argv: str, typed=()) -> Run:
        """Run `wildhat <argv>`, typing `typed` at its prompts.

        An entry in `typed` can be a function, called at the moment wildhat
        asks, for input that depends on what has happened so far.
        """
        self.typed = list(typed)
        out, err = io.StringIO(), io.StringIO()
        self.output = out
        result = {}

        def run():
            try:
                result["code"] = cli.main(list(argv))
            except SystemExit as e:  # argparse errors and --version
                result["code"] = e.code
            except BaseException as e:
                result["error"] = e

        # Anything wildhat leaves open (an HTTP error's body, say) is reported
        # when it's garbage-collected, which can be at some random later
        # moment; collect at the end, so it's always caught in the run that
        # leaked it. (Warning filters are process-wide, so this covers the
        # run's thread too.)
        with warnings.catch_warnings(record=True) as leaks:
            warnings.simplefilter("always", ResourceWarning)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                thread = threading.Thread(target=run, daemon=True)
                thread.start()
                thread.join(RUN_TIMEOUT)
            if not thread.is_alive():
                gc.collect()
        if thread.is_alive():
            self.fail(f"wildhat {' '.join(argv)} hung.\nOutput so far:\n{out.getvalue()}{err.getvalue()}")
        if "error" in result:
            raise result["error"]
        leaks = [w for w in leaks if issubclass(w.category, ResourceWarning)]
        if leaks:
            self.fail(f"wildhat {' '.join(argv)} left something open: {leaks[0].message}")
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

    def gcloud_login(self, *accounts: str, active: str | None = None) -> None:
        """Sign Google user accounts in to the fake gcloud, as `gcloud auth
        login` would. The last one is active unless `active` says otherwise."""
        path = self.fake_dir / "gcloud.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "config": {}, "accounts": {}, "active": None
        }
        for account in accounts:
            state["accounts"][account] = {"cred_file": None}
        # Mentors use gcloud for other projects too, so wildhat can't rely on
        # its default project.
        state["config"].setdefault("project", "some-other-project")
        state["active"] = active or accounts[-1]
        path.write_text(json.dumps(state), encoding="utf-8")

    def gcloud_state(self) -> dict:
        return json.loads((self.fake_dir / "gcloud.json").read_text(encoding="utf-8"))

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
