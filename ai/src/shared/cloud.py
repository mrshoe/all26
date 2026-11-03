"""Reading the team's bucket and Firestore from code in ai/src, the same way
on a laptop and in a `wildhat launch` job:

    cloud = Cloud()
    cloud.document("runs/20261102-0019-w22f")      {"launched_by": "kai-morales", ...}
    cloud.documents("runs/20261102-0019-w22f/episodes")
    cloud.objects("runs/kai-morales/20261102-0019-w22f/")   [("runs/.../episode.json", 812), ...]
    cloud.download("runs/.../episode.json", Path("episode.json"))

On a laptop it uses `wildhat access-token`, so you need to have run
`wildhat login` (and have wildhat 0.4.0 or newer). In a cloud job it uses the
job's own service account, from the metadata server. Either way it acts as
you: everyone can read the whole bucket and every record.

Only the standard library, so it works anywhere ai/src runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STORAGE = "https://storage.googleapis.com"
FIRESTORE = "https://firestore.googleapis.com/v1"
METADATA = "http://metadata.google.internal/computeMetadata/v1"
# Google's access tokens last an hour; get a new one well before then.
TOKEN_SECONDS = 45 * 60


class CloudError(Exception):
    pass


def in_cloud_job() -> bool:
    """True inside a `wildhat launch` job (the base image's entrypoint sets
    WILDHAT_OUTPUT)."""
    return "WILDHAT_OUTPUT" in os.environ


def plain(value: dict):
    """A Firestore typed value ({"integerValue": "3"}) as plain Python.
    Timestamps stay as their RFC 3339 strings."""
    (kind, v), = value.items()
    if kind == "integerValue":
        return int(v)
    if kind == "arrayValue":
        return [plain(x) for x in v.get("values", [])]
    if kind == "mapValue":
        return {k: plain(x) for k, x in v.get("fields", {}).items()}
    if kind == "nullValue":
        return None
    return v


class Cloud:
    def __init__(self):
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires = 0.0
        self.project: str | None = None
        self.bucket: str | None = None

    # --- Credentials ------------------------------------------------------

    def token(self) -> str:
        with self._lock:
            if self._token is None or time.time() > self._expires:
                self._token, self.project, self.bucket = self._from_job() if in_cloud_job() else self._from_wildhat()
                self._expires = time.time() + TOKEN_SECONDS
            return self._token

    def _from_job(self) -> tuple[str, str, str]:
        def metadata(path: str) -> bytes:
            req = urllib.request.Request(f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read()

        token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
        project = metadata("project/project-id").decode()
        # gs://<bucket>/runs/<handle>/<run>/
        bucket = os.environ["WILDHAT_OUTPUT"].removeprefix("gs://").split("/")[0]
        return token, project, bucket

    def _from_wildhat(self) -> tuple[str, str, str]:
        wildhat = shutil.which("wildhat")
        if not wildhat:
            raise CloudError("Can't find the wildhat command. Install it (ai/wildhat/README.md), "
                             "then run wildhat login.")
        result = subprocess.run([wildhat, "access-token"], capture_output=True, text=True)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            if "invalid choice: 'access-token'" in message:
                raise CloudError("Your wildhat is too old for this; upgrade it (ai/wildhat/README.md, \"Upgrade\").")
            raise CloudError(f"Couldn't get your credentials from wildhat: {message} "
                             "(Have you run wildhat login?)")
        reply = json.loads(result.stdout)
        return reply["access_token"], reply["project"], reply["bucket"]

    # --- Requests -----------------------------------------------------------

    def _open(self, url: str, method="GET", body: dict | None = None, timeout=120):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            with e:  # Close the error's body, or Python warns later.
                try:
                    message = json.load(e)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = e.reason
            raise CloudError(f"Google said (HTTP {e.code}): {message}") from None
        except urllib.error.URLError as e:
            raise CloudError(f"Couldn't reach Google ({e.reason}). Are you online?") from None

    def _json(self, url: str, method="GET", body: dict | None = None):
        with self._open(url, method, body) as resp:
            text = resp.read()
            return json.loads(text) if text else {}

    def _firestore(self) -> str:
        self.token()
        return f"{FIRESTORE}/projects/{self.project}/databases/(default)/documents"

    # --- Firestore ------------------------------------------------------------

    def document(self, path: str) -> dict | None:
        """A record as plain Python, or None if there's no such record."""
        try:
            doc = self._json(f"{self._firestore()}/{path}")
        except CloudError as e:
            if "HTTP 404" in str(e):
                return None
            raise
        return {k: plain(v) for k, v in doc.get("fields", {}).items()}

    def documents(self, collection: str) -> dict[str, dict]:
        """Every record in a collection, by id."""
        out, page = {}, ""
        while True:
            query = urllib.parse.urlencode({"pageSize": 300, "pageToken": page})
            reply = self._json(f"{self._firestore()}/{collection}?{query}")
            for doc in reply.get("documents", []):
                out[doc["name"].rsplit("/", 1)[1]] = {k: plain(v) for k, v in doc.get("fields", {}).items()}
            page = reply.get("nextPageToken")
            if not page:
                return out

    # --- The bucket -------------------------------------------------------------

    def objects(self, prefix: str) -> list[tuple[str, int]]:
        """Every object under a prefix, as (name, size in bytes)."""
        self.token()
        out, page = [], ""
        while True:
            query = urllib.parse.urlencode({"prefix": prefix, "pageToken": page,
                                            "fields": "items(name,size),nextPageToken"})
            reply = self._json(f"{STORAGE}/storage/v1/b/{self.bucket}/o?{query}")
            out += [(item["name"], int(item.get("size", 0))) for item in reply.get("items", [])]
            page = reply.get("nextPageToken")
            if not page:
                return out

    def download(self, name: str, path: Path) -> None:
        """Save one object to path. A partial download never takes path's
        place, so an interrupted download is simply done again."""
        self.token()
        path.parent.mkdir(parents=True, exist_ok=True)
        quoted = urllib.parse.quote(name, safe="")
        partial = path.with_name(path.name + ".part")
        with self._open(f"{STORAGE}/storage/v1/b/{self.bucket}/o/{quoted}?alt=media", timeout=300) as resp, \
                partial.open("wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)
        os.replace(partial, path)
