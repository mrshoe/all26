"""The visualizer's server: the API the browser app calls, and the app itself.

    GET /api/config                         what the sign-in page needs (public)
    GET /api/me                             who's signed in
    GET /api/runs?user=&sim=&limit=         runs, newest first
    GET /api/runs/<run>                     a run and its episodes
    GET /api/runs/<run>/episodes/<n>        an episode's episode.json, tables,
                                            and links to its preview videos

Everything under /api but /api/config needs a Firebase ID token, from the
same email-link sign-in as `wildhat login`. Only people on the roster have a
`handle` claim (`wildhat admin sync` sets it), so that's the allowlist.

Videos don't pass through here: the API hands out signed links, good for an
hour, and the browser plays them straight from the bucket.

It reads Firestore and the bucket as its own service account, which can read
but not change them (../../terraform/visualizer.tf).
"""

# No `from __future__ import annotations`: FastAPI reads the endpoints'
# annotations, and some name types defined inside create_app.
import datetime
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Optional, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ID = os.environ.get("PROJECT_ID", "")
BUCKET = os.environ.get("BUCKET", f"{PROJECT_ID}-wildhat")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
WEB = Path(os.environ.get("WEB_DIR", Path(__file__).resolve().parent.parent / "web" / "dist"))
SIGNED_URL_SECONDS = 60 * 60
MAX_RUNS = 200


# ---------------------------------------------------------------------------
# Where the data is. Tests swap in their own.
# ---------------------------------------------------------------------------


class Store(Protocol):
    def runs(self, user: Optional[str], limit: int) -> list[dict]: ...

    def run(self, run_id: str) -> Optional[dict]: ...

    def episodes(self, run_id: str) -> list[dict]: ...

    def read(self, path: str) -> Optional[bytes]:
        """An object in the bucket, or None if it isn't there."""

    def signed_url(self, path: str, content_type: str) -> str: ...


class GoogleStore:
    def __init__(self):
        import google.auth
        from google.cloud import firestore, storage

        self.credentials, _ = google.auth.default()
        self.db = firestore.Client(project=PROJECT_ID)
        self.bucket = storage.Client(project=PROJECT_ID).bucket(BUCKET)
        self.lock = threading.Lock()

    def runs(self, user, limit):
        from google.cloud import firestore

        query = self.db.collection("runs")
        if user:
            query = query.where(filter=firestore.FieldFilter("launched_by", "==", user))
        query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
        return [{"id": doc.id, **doc.to_dict()} for doc in query.stream()]

    def run(self, run_id):
        doc = self.db.collection("runs").document(run_id).get()
        return {"id": doc.id, **doc.to_dict()} if doc.exists else None

    def episodes(self, run_id):
        docs = self.db.collection("runs").document(run_id).collection("episodes").stream()
        return sorted((doc.to_dict() for doc in docs), key=lambda e: e.get("index", 0))

    def read(self, path):
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(path).download_as_bytes()
        except NotFound:
            return None

    def signed_url(self, path, content_type):
        from google.auth.transport.requests import Request

        # Cloud Run has no private key to sign with, so IAM signs for us
        # (the service account may sign as itself), given an access token.
        with self.lock:
            if not self.credentials.valid:
                self.credentials.refresh(Request())
            token = self.credentials.token
        return self.bucket.blob(path).generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=SIGNED_URL_SECONDS),
            method="GET",
            service_account_email=self.credentials.service_account_email,
            access_token=token,
            # Runs upload everything as application/octet-stream, which
            # Safari won't play.
            response_type=content_type,
        )


def verify_firebase_token(token: str) -> dict:
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    return id_token.verify_firebase_token(token, Request(), audience=PROJECT_ID)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class Person(BaseModel):
    email: str
    handle: str


class Config(BaseModel):
    project_id: str
    firebase_api_key: str


def create_app(store: Store, verify=verify_firebase_token) -> FastAPI:
    app = FastAPI(title="Wildhat visualizer", docs_url="/api/docs", openapi_url="/api/openapi.json")
    verified: dict[str, dict] = {}
    pool = ThreadPoolExecutor(8)

    def person(authorization: Annotated[Optional[str], Header()] = None) -> Person:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Sign in first.")
        token = authorization.removeprefix("Bearer ")
        claims = verified.get(token)
        if claims is None or claims["exp"] < time.time():
            try:
                claims = verify(token)
            except Exception:
                raise HTTPException(401, "Your sign-in has expired. Sign in again.") from None
            if len(verified) > 1000:
                verified.clear()
            verified[token] = claims
        if not claims.get("email_verified") or not claims.get("handle"):
            raise HTTPException(403, "You're not on the team roster. Ask a mentor.")
        return Person(email=claims["email"], handle=claims["handle"])

    Signed = Annotated[Person, Depends(person)]

    @app.get("/api/config")
    def config() -> Config:
        return Config(project_id=PROJECT_ID, firebase_api_key=FIREBASE_API_KEY)

    @app.get("/api/me")
    def me(who: Signed) -> Person:
        return who

    @app.get("/api/runs")
    def runs(
        who: Signed,
        user: Annotated[Optional[str], Query(description="only this person's runs")] = None,
        sim: Annotated[bool, Query(description="only simulation runs")] = False,
        limit: Annotated[int, Query(ge=1, le=MAX_RUNS)] = 50,
    ) -> list[dict]:
        found = store.runs(user or None, limit)
        if sim:
            found = [r for r in found if r.get("sim")]
        summaries = pool.map(lambda r: summarize(r, store.episodes(r["id"])), found)
        return [plain(s) for s in summaries]

    @app.get("/api/runs/{run_id}")
    def run(who: Signed, run_id: str) -> dict:
        found = store.run(run_id)
        if not found:
            raise HTTPException(404, f"There's no run {run_id}.")
        episodes = store.episodes(run_id)
        return plain({**summarize(found, episodes), "episodes_detail": episodes})

    @app.get("/api/runs/{run_id}/episodes/{index}")
    def episode(who: Signed, run_id: str, index: int) -> dict:
        found = store.run(run_id)
        if not found:
            raise HTTPException(404, f"There's no run {run_id}.")
        prefix = episode_prefix(found, index)
        info = store.read(prefix + "episode.json")
        if info is None:
            raise HTTPException(404, f"Episode {index} of {run_id} has no episode.json (yet).")
        info = json.loads(info)
        tables: dict[str, dict] = {}
        if info.get("format", 1) >= 2:
            # Pictures' paths aren't needed: the browser plays the previews.
            names = [t for t in info.get("tables", []) if t != "images"]
            found = pool.map(lambda t: store.read(f"{prefix}{t}.parquet"), names)
            tables = {t: read_table(data) for t, data in zip(names, found) if data is not None}
            cameras = camera_names(tables)
        else:
            # Format 1, from before the tables: only its videos can be shown.
            cameras = list(info.get("cameras", {}))
        previews = {
            camera: {
                "plain": store.signed_url(f"{prefix}preview/{camera}.mp4", "video/mp4"),
                "boxes": store.signed_url(f"{prefix}preview/{camera}.boxes.mp4", "video/mp4"),
            }
            for camera in cameras
        }
        return {
            "run": run_id,
            "index": index,
            "format": info.get("format", 1),
            "info": info,
            "tables": tables,
            "previews": previews,
        }

    if (WEB / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def web(path: str):
            # The app routes in the browser, so every page is index.html.
            return FileResponse(WEB / "index.html")

    return app


def episode_prefix(run: dict, index: int) -> str:
    """The bucket path of an episode's outputs, from the run's gs:// prefix."""
    outputs = run.get("outputs", "")
    if not outputs.startswith(f"gs://{BUCKET}/"):
        raise HTTPException(404, "This run's outputs aren't in the wildhat bucket.")
    return outputs.removeprefix(f"gs://{BUCKET}/").rstrip("/") + f"/episode-{index}/"


def summarize(run: dict, episodes: list[dict]) -> dict:
    """The run, with how many of its episodes are in each state. Episodes
    record their own state, so one cancelled before it started stays
    "pending"; `wildhat runs` works out the true state from Cloud Run."""
    counts: dict[str, int] = {}
    for e in episodes:
        counts[e.get("status", "pending")] = counts.get(e.get("status", "pending"), 0) + 1
    return {**run, "status_counts": counts}


def read_table(data: bytes) -> dict[str, list]:
    """A parquet table as {column: values}, NaN as null."""
    import io

    import pyarrow.parquet as pq

    columns = pq.read_table(io.BytesIO(data)).to_pydict()
    return {k: [None if isinstance(v, float) and math.isnan(v) else v for v in vs] for k, vs in columns.items()}


def camera_names(tables: dict[str, dict]) -> list[str]:
    """The cameras' names, from the cameras and objects tables."""
    if "cameras" not in tables or "objects" not in tables:
        return []
    names = dict(zip(tables["objects"]["object"], tables["objects"]["name"]))
    return [names[i] for i in tables["cameras"]["object"] if i in names]


def plain(value: Any) -> Any:
    """Firestore's values as JSON: timestamps as ISO strings."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value


def main() -> FastAPI:
    return create_app(GoogleStore())
