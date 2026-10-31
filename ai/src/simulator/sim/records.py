"""Adding what the visualizer needs to the run's Firestore records.

`wildhat launch` creates runs/<run> and runs/<run>/episodes/<n>, and the
base image's entrypoint keeps each episode's status. Here, each episode adds
a `sim` field to both: the run's says which scenario ran (so the visualizer
can list and filter simulation runs), and the episode's says what it made.
Every episode writes the same run fields, so it doesn't matter which is first.

Only in the cloud (WILDHAT_RUN_DOC is set), and best effort, like the
entrypoint: a Firestore problem is printed but never fails the episode.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request

METADATA_TOKEN = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)
FIRESTORE = "https://firestore.googleapis.com/v1"


def value(v) -> dict:
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, float):
        return {"nullValue": None} if math.isnan(v) else {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [value(x) for x in v]}}
    if isinstance(v, dict):
        return {"mapValue": {"fields": {str(k): value(x) for k, x in v.items()}}}
    if v is None:
        return {"nullValue": None}
    raise TypeError(f"Can't store {v!r} in Firestore")


def patch(doc: str, fields: dict) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"}), timeout=10
    ) as resp:
        token = json.load(resp)["access_token"]
    mask = "&".join(f"updateMask.fieldPaths={name}" for name in fields)
    body = json.dumps({"fields": {k: value(v) for k, v in fields.items()}}).encode()
    req = urllib.request.Request(
        f"{FIRESTORE}/{doc}?{mask}", data=body, method="PATCH",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30):
        pass


def record(episode: dict, previews: list[str]) -> None:
    run_doc = os.environ.get("WILDHAT_RUN_DOC")
    if not run_doc:
        return
    index = os.environ.get("WILDHAT_EPISODE", "0")
    run = {"scenario": episode["scenario"], "scene": episode["scene"], "backend": episode["backend"]}
    mine = {
        **run,
        "seed": episode["seed"],
        "frames": episode["frames"],
        "fps": episode["fps"],
        "cameras": list(episode["cameras"]),
        "labels": sorted(set(episode["labels"].values())),
        "previews": previews,
    }
    try:
        patch(run_doc, {"sim": run})
        patch(f"{run_doc}/episodes/{index}", {"sim": mine})
    except OSError as e:  # urllib's errors are OSErrors
        print(f"sim: couldn't record the episode in Firestore: {e}", flush=True)
