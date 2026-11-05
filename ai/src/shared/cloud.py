"""Reading the team's bucket and Firestore from code in ai/src, the same way
on a laptop and in a `wildhat launch` job:

    cloud = Cloud()
    cloud.document("runs/20261102-0019-w22f")      {"launched_by": "kai-morales", ...}
    cloud.documents("runs/20261102-0019-w22f/episodes")
    cloud.objects("runs/kai-morales/20261102-0019-w22f/")   [("runs/.../episode.json", 812), ...]
    cloud.download("runs/.../episode.json", Path("episode.json"))
    cloud.create("training/<id>", {...}); cloud.update("training/<id>", {...})
    cloud.upload("training/<handle>/<id>/metrics/000001.jsonl", b"...")
    cloud.handle                                     "kai-morales": who you are

On a laptop it uses `wildhat access-token`, so you need to have run
`wildhat login` (and have wildhat 0.4.0 or newer). In a cloud job it uses the
job's own service account, from the metadata server. Either way it acts as
you: everyone can read the whole bucket and every record.

Only the standard library, so it works anywhere ai/src runs.
"""

from __future__ import annotations

import datetime
import json
import os
import random
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
# Reads that fail for a reason that may pass (the network, Google busy) are
# tried this many times in all, waiting 1, 2, 4, 8 s (and a little) between.
TRIES = 5
TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


class CloudError(Exception):
    pass


class TransientCloudError(CloudError):
    """A failure that may well not happen if tried again."""


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


def firestore_value(value) -> dict:
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, datetime.datetime):
        return {"timestampValue": value.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [firestore_value(v) for v in value]}}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {str(k): firestore_value(v) for k, v in value.items()}}}
    if value is None:
        return {"nullValue": None}
    raise TypeError(f"Can't store {value!r} in Firestore")


def patch_body(fields: dict) -> tuple[list[str], dict]:
    """A record update's field mask, and its fields in Firestore's typed form.
    A dotted name like "best.step" sets a field inside a map."""
    nested: dict = {}
    for name, value in fields.items():
        *parents, last = name.split(".")
        where = nested
        for parent in parents:
            where = where.setdefault(parent, {})
        where[last] = value
    return list(fields), firestore_value(nested)["mapValue"]["fields"]


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Cloud:
    def __init__(self):
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires = 0.0
        self.project: str | None = None
        self.bucket: str | None = None
        self._handle: str | None = None

    @property
    def handle(self) -> str:
        """Who this is acting as, e.g. kai-morales."""
        self.token()
        return self._handle

    # --- Credentials ------------------------------------------------------

    def token(self) -> str:
        with self._lock:
            if self._token is None or time.time() > self._expires:
                self._token, self.project, self.bucket, self._handle = (
                    self._from_job() if in_cloud_job() else self._from_wildhat())
                self._expires = time.time() + TOKEN_SECONDS
            return self._token

    def _from_job(self) -> tuple[str, str, str, str]:
        def metadata(path: str) -> bytes:
            req = urllib.request.Request(f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read()

        token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
        project = metadata("project/project-id").decode()
        # gs://<bucket>/runs/<handle>/<run>/: jobs run as the person who launched them.
        bucket, _, handle = os.environ["WILDHAT_OUTPUT"].removeprefix("gs://").split("/")[:3]
        return token, project, bucket, handle

    def _from_wildhat(self) -> tuple[str, str, str, str]:
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
        return reply["access_token"], reply["project"], reply["bucket"], reply["handle"]

    # --- Requests -----------------------------------------------------------

    def _open(self, url: str, method="GET", body=None, timeout=120, content_type="application/json",
              length: int | None = None):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        headers = {"Authorization": f"Bearer {self.token()}", "Content-Type": content_type}
        if length is not None:
            headers["Content-Length"] = str(length)
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            with e:  # Close the error's body, or Python warns later.
                try:
                    message = json.load(e)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = e.reason
            error = TransientCloudError if e.code in TRANSIENT_HTTP else CloudError
            raise error(f"Google said (HTTP {e.code}): {message}") from None
        except urllib.error.URLError as e:
            raise TransientCloudError(f"Couldn't reach Google ({e.reason}). Are you online?") from None
        except (ConnectionError, TimeoutError) as e:
            raise TransientCloudError(f"Lost the connection to Google ({e}). Are you online?") from None

    @staticmethod
    def _retrying(attempt):
        """attempt(), tried again after failures that may pass. Only for
        requests that do the same thing however often they're made."""
        for n in range(TRIES):
            try:
                return attempt()
            except TransientCloudError:
                if n == TRIES - 1:
                    raise
                time.sleep(2 ** n + random.random())
            except (ConnectionError, TimeoutError) as e:  # while reading the reply
                if n == TRIES - 1:
                    raise TransientCloudError(f"Lost the connection to Google ({e}). Are you online?") from None
                time.sleep(2 ** n + random.random())

    def _json(self, url: str, method="GET", body: dict | None = None):
        def attempt():
            with self._open(url, method, body) as resp:
                text = resp.read()
                return json.loads(text) if text else {}
        return self._retrying(attempt) if method == "GET" else attempt()

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

    def create(self, path: str, fields: dict) -> None:
        """A new record; fails if there's one already."""
        collection, doc_id = path.rsplit("/", 1)
        query = urllib.parse.urlencode({"documentId": doc_id})
        self._json(f"{self._firestore()}/{collection}?{query}", "POST",
                   {"fields": firestore_value(fields)["mapValue"]["fields"]})

    def update(self, path: str, fields: dict) -> None:
        """Set these fields of a record (creating it if need be), leaving its
        others alone. A dotted name like "best.step" sets a field inside a map."""
        self.patch(path, *patch_body(fields))

    def patch(self, path: str, mask: list[str], typed: dict) -> None:
        """update() with the fields already in Firestore's typed form, as
        patch_body() makes them (and the training upload queue stores them)."""
        query = "&".join("updateMask.fieldPaths=" + urllib.parse.quote(name) for name in mask)
        self._json(f"{self._firestore()}/{path}?{query}", "PATCH", {"fields": typed})

    def delete_document(self, path: str) -> None:
        with self._open(f"{self._firestore()}/{path}", "DELETE"):
            pass

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

    def upload(self, name: str, data: bytes | Path, content_type="application/octet-stream") -> None:
        """Save bytes, or a file (streamed, so big checkpoints are fine), as
        an object, replacing any there."""
        self.token()
        query = urllib.parse.urlencode({"uploadType": "media", "name": name})
        url = f"{STORAGE}/upload/storage/v1/b/{self.bucket}/o?{query}"
        if isinstance(data, Path):
            with data.open("rb") as f, self._open(url, "POST", f, 600, content_type, data.stat().st_size):
                pass
        else:
            with self._open(url, "POST", data, 120, content_type):
                pass

    def delete(self, name: str) -> None:
        self.token()
        quoted = urllib.parse.quote(name, safe="")
        with self._open(f"{STORAGE}/storage/v1/b/{self.bucket}/o/{quoted}", "DELETE"):
            pass

    def download(self, name: str, path: Path) -> None:
        """Save one object to path. A partial download never takes path's
        place, so an interrupted download is simply done again."""
        self.token()
        path.parent.mkdir(parents=True, exist_ok=True)
        quoted = urllib.parse.quote(name, safe="")
        partial = path.with_name(path.name + ".part")

        def attempt():
            with self._open(f"{STORAGE}/storage/v1/b/{self.bucket}/o/{quoted}?alt=media", timeout=300) as resp, \
                    partial.open("wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
        self._retrying(attempt)
        os.replace(partial, path)
