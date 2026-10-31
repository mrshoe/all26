"""The visualizer's API, with an in-memory Firestore and bucket. Run from
ai/visualizer/server:

    python -m unittest discover -s tests
"""

import datetime
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path

os.environ.update(PROJECT_ID="test-project", BUCKET="test-bucket", FIREBASE_API_KEY="test-key",
                  WEB_DIR="/nonexistent")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402

KAI = {"email": "813579@seq.org", "email_verified": True, "handle": "kai-morales", "exp": time.time() + 3600}
STRANGER = {"email": "someone@example.com", "email_verified": True, "exp": time.time() + 3600}
TOKENS = {"kai": KAI, "stranger": STRANGER}
WHEN = datetime.datetime(2026, 10, 30, 21, 30, tzinfo=datetime.timezone.utc)


def verify(token):
    if token not in TOKENS:
        raise ValueError("bad token")
    return TOKENS[token]


class FakeStore:
    def __init__(self):
        self.docs = {
            "r1": {"launched_by": "kai-morales", "created_at": WHEN, "outputs": "gs://test-bucket/runs/kai-morales/r1/",
                   "sim": {"scenario": "scenarios/yellow_ball.py:YellowBall", "scene": "Gym", "backend": "blender"}},
            "r2": {"launched_by": "dana-wells", "created_at": WHEN - datetime.timedelta(hours=1),
                   "outputs": "gs://test-bucket/runs/dana-wells/r2/"},
        }
        self.episode_docs = {"r1": [{"index": 0, "status": "succeeded"}, {"index": 1, "status": "running"}], "r2": []}
        self.objects = {}
        signals = io.BytesIO()
        pq.write_table(pa.table({"frame": [0, 1], "ball.z": [0.5, float("nan")]}), signals)
        prefix = "runs/kai-morales/r1/episode-0/"
        self.objects[prefix + "episode.json"] = json.dumps({"cameras": {"front": {}}, "frames": 2}).encode()
        self.objects[prefix + "signals.parquet"] = signals.getvalue()

    def runs(self, user, limit):
        found = [{"id": k, **v} for k, v in self.docs.items() if not user or v["launched_by"] == user]
        return sorted(found, key=lambda r: r["created_at"], reverse=True)[:limit]

    def run(self, run_id):
        return {"id": run_id, **self.docs[run_id]} if run_id in self.docs else None

    def episodes(self, run_id):
        return self.episode_docs.get(run_id, [])

    def read(self, path):
        return self.objects.get(path)

    def signed_url(self, path, content_type):
        return f"https://signed.example/{path}?type={content_type}"


class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.create_app(FakeStore(), verify))

    def get(self, path, token="kai"):
        return self.client.get(path, headers={"Authorization": f"Bearer {token}"} if token else {})

    def test_config_is_public(self):
        self.assertEqual(self.get("/api/config", token=None).json(),
                         {"project_id": "test-project", "firebase_api_key": "test-key"})

    def test_sign_in_is_required(self):
        self.assertEqual(self.get("/api/runs", token=None).status_code, 401)
        self.assertEqual(self.get("/api/runs", token="forged").status_code, 401)

    def test_only_people_on_the_roster(self):
        response = self.get("/api/me", token="stranger")
        self.assertEqual(response.status_code, 403)
        self.assertIn("roster", response.json()["detail"])
        self.assertEqual(self.get("/api/me").json(), {"email": "813579@seq.org", "handle": "kai-morales"})

    def test_runs(self):
        runs = self.get("/api/runs").json()
        self.assertEqual([r["id"] for r in runs], ["r1", "r2"])
        self.assertEqual(runs[0]["created_at"], "2026-10-30T21:30:00+00:00")
        self.assertEqual(runs[0]["status_counts"], {"succeeded": 1, "running": 1})
        self.assertEqual([r["id"] for r in self.get("/api/runs?user=dana-wells").json()], ["r2"])
        self.assertEqual([r["id"] for r in self.get("/api/runs?sim=true").json()], ["r1"])

    def test_a_run(self):
        run = self.get("/api/runs/r1").json()
        self.assertEqual(run["sim"]["scene"], "Gym")
        self.assertEqual([e["index"] for e in run["episodes_detail"]], [0, 1])
        self.assertEqual(self.get("/api/runs/nope").status_code, 404)

    def test_an_episode(self):
        ep = self.get("/api/runs/r1/episodes/0").json()
        self.assertEqual(ep["info"]["frames"], 2)
        self.assertEqual(ep["signals"], {"frame": [0, 1], "ball.z": [0.5, None]})
        self.assertEqual(ep["previews"]["front"], {
            "plain": "https://signed.example/runs/kai-morales/r1/episode-0/preview/front.mp4?type=video/mp4",
            "boxes": "https://signed.example/runs/kai-morales/r1/episode-0/preview/front.boxes.mp4?type=video/mp4",
        })

    def test_an_episode_without_data_yet(self):
        response = self.get("/api/runs/r1/episodes/1")
        self.assertEqual(response.status_code, 404)
        self.assertIn("no episode.json", response.json()["detail"])

    def test_outputs_elsewhere_are_refused(self):
        store = FakeStore()
        store.docs["r1"]["outputs"] = "gs://another-bucket/x/"
        client = TestClient(app.create_app(store, verify))
        response = client.get("/api/runs/r1/episodes/0", headers={"Authorization": "Bearer kai"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
