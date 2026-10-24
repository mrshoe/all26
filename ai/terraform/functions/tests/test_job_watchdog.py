"""Tests for the job watchdog function (../job_watchdog/main.py).

They call the real check_jobs() and fake only what's outside it: Google's APIs
(the metadata server, BigQuery and Cloud Run), by replacing urlopen. The tests
live outside job_watchdog/ so Terraform doesn't deploy them.

Run from ai/terraform/functions: python -m unittest discover -s tests
"""

import contextlib
import datetime
import http.client
import importlib.util
import io
import json
import os
import sys
import types
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

PROJECT = "test-project"
REGION = "us-central1"
TABLE = f"{PROJECT}.billing_export.gcp_billing_export_v1_X"
RUN = "https://run.googleapis.com/v2"
TOKEN = "watchdog-token"
# The same limits as ai/terraform/guardrails.tf.
os.environ.update(
    PROJECT=PROJECT,
    REGION=REGION,
    MAX_RUNTIME_HOURS="8",
    BUDGET_USD="300",
    BILLING_TABLE=TABLE,
    MAX_GPUS_PER_USER="5",
    MAX_GPUS="10",
)
# The Functions Framework only matters when deployed; its decorator just
# registers the function.
sys.modules["functions_framework"] = types.SimpleNamespace(http=lambda f: f)
MAIN = Path(__file__).resolve().parents[1] / "job_watchdog" / "main.py"
spec = importlib.util.spec_from_file_location("job_watchdog_main", MAIN)
watchdog = importlib.util.module_from_spec(spec)
# Leave no __pycache__ in the folder Terraform deploys.
sys.dont_write_bytecode = True
spec.loader.exec_module(watchdog)

NOW = datetime.datetime.now(datetime.timezone.utc)


def stamp(minutes_ago: float) -> str:
    t = NOW - datetime.timedelta(minutes=minutes_ago)
    # Cloud Run's format, nanoseconds and all.
    return t.strftime("%Y-%m-%dT%H:%M:%S.123456789Z")


def execution(
    job: str,
    name: str,
    *,
    creator: str = "wh-alice@test-project.iam.gserviceaccount.com",
    minutes_ago: float = 5,
    gpus: int = 0,
    tasks: int = 1,
    parallelism: int | None = None,
    done: bool = False,
    **counts,
) -> dict:
    limits = {"cpu": "4", "memory": "16Gi"}
    if gpus:
        limits["nvidia.com/gpu"] = str(gpus)
    e = {
        "name": f"projects/{PROJECT}/locations/{REGION}/jobs/{job}/executions/{name}",
        "creator": creator,
        "createTime": stamp(minutes_ago),
        "startTime": stamp(minutes_ago - 0.1),
        "taskCount": tasks,
        "template": {
            "containers": [{"image": "us-docker.pkg.dev/x/y", "resources": {"limits": limits}}],
            **({"nodeSelector": {"accelerator": "nvidia-l4"}} if gpus else {}),
        },
        **counts,
    }
    if parallelism is not None:
        e["parallelism"] = parallelism
    if done:
        e["completionTime"] = stamp(0)
    return e


class FakeCloud:
    """The metadata server, BigQuery and Cloud Run. Unknown calls fail the test."""

    PAGE = 2  # Small pages, so paging gets tested.

    def __init__(self):
        self.executions: list[dict] = []
        self.cost: float | None = 10.0
        self.billing = "ok"  # or "missing", "error", "slow"
        self.cancelled: list[str] = []
        self.queries: list[dict] = []

    def urlopen(self, req, timeout=None):
        url = urllib.parse.urlsplit(req.full_url)
        query = dict(urllib.parse.parse_qsl(url.query))
        if url.hostname == "metadata.google.internal":
            assert req.get_header("Metadata-flavor") == "Google"
            return self._reply({"access_token": TOKEN, "expires_in": 3599})
        assert req.get_header("Authorization") == f"Bearer {TOKEN}", req.full_url
        if url.hostname == "bigquery.googleapis.com":
            assert url.path == f"/bigquery/v2/projects/{PROJECT}/queries"
            body = json.loads(req.data)
            self.queries.append(body)
            if self.billing == "missing":
                return self._error(req.full_url, 404, f"Not found: Table {TABLE}")
            if self.billing == "error":
                return self._error(req.full_url, 500, "Backend error")
            if self.billing == "slow":
                return self._reply({"jobComplete": False})
            value = None if self.cost is None else str(self.cost)
            return self._reply({"jobComplete": True, "rows": [{"f": [{"v": value}]}]})
        if url.hostname == "run.googleapis.com":
            path = url.path[len("/v2/"):]
            if req.get_method() == "POST" and path.endswith(":cancel"):
                self.cancelled.append(path[: -len(":cancel")])
                return self._reply({"name": path})
            if path == f"projects/{PROJECT}/locations/{REGION}/jobs":
                jobs = sorted({e["name"].split("/executions/")[0] for e in self.executions})
                return self._page([{"name": j} for j in jobs], "jobs", query)
            if path.endswith("/executions"):
                job = path[: -len("/executions")]
                mine = [e for e in self.executions if e["name"].startswith(job + "/executions/")]
                # Cloud Run lists the newest first.
                mine.sort(key=lambda e: e["createTime"], reverse=True)
                return self._page(mine, "executions", query)
        raise AssertionError(f"Unexpected request: {req.get_method()} {req.full_url}")

    def _page(self, items: list, field: str, query: dict):
        start = int(query.get("pageToken") or 0)
        reply = {field: items[start : start + self.PAGE]}
        if start + self.PAGE < len(items):
            reply["nextPageToken"] = str(start + self.PAGE)
        return self._reply(reply)

    def _reply(self, body: dict):
        response = io.BytesIO(json.dumps(body).encode())
        response.headers = http.client.HTTPMessage()
        return response

    def _error(self, url: str, status: int, message: str):
        body = json.dumps({"error": {"code": status, "message": message}}).encode()
        raise urllib.error.HTTPError(url, status, message, http.client.HTTPMessage(), io.BytesIO(body))


class Request:
    """What the Functions Framework hands check_jobs."""

    def __init__(self, **args):
        self.args = args


class WatchdogTestCase(unittest.TestCase):
    def setUp(self):
        self.cloud = FakeCloud()
        patcher = mock.patch("urllib.request.urlopen", self.cloud.urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, **args) -> dict:
        self.log = io.StringIO()
        with contextlib.redirect_stdout(self.log):
            result = watchdog.check_jobs(Request(**args))
        json.dumps(result)  # The Functions Framework sends it back as JSON.
        self.assertEqual(sorted(result["cancelled"]), sorted(self.cloud.cancelled))
        self.assertEqual(len(set(self.cloud.cancelled)), len(self.cloud.cancelled), "cancelled twice")
        return result

    def cancelled(self) -> list[str]:
        return sorted(name.split("/executions/")[1] for name in self.cloud.cancelled)


class RuntimeLimitTest(WatchdogTestCase):
    def test_nothing_running(self):
        result = self.check()

        self.assertEqual(result, {"month_cost": 10.0, "checked": 0, "cancelled": []})

    def test_cancels_executions_running_over_8_hours(self):
        self.cloud.executions = [
            execution("train", "weekend", minutes_ago=8 * 60 + 5),
            execution("train", "today", minutes_ago=7 * 60),
            execution("eval", "finished-long-ago", minutes_ago=20 * 60, done=True),
        ]

        result = self.check()

        self.assertEqual(self.cancelled(), ["weekend"])
        self.assertEqual(result["checked"], 3)
        self.assertIn("Running longer than 8:00:00", self.log.getvalue())

    def test_executions_not_started_yet_count_from_creation(self):
        e = execution("train", "stuck", minutes_ago=9 * 60)
        del e["startTime"]
        self.cloud.executions = [e]

        self.check()

        self.assertEqual(self.cancelled(), ["stuck"])

    def test_kill_switch_cancels_everything_running(self):
        self.cloud.executions = [
            execution("train", "a", minutes_ago=1),
            execution("train", "b", minutes_ago=0.5),
            execution("eval", "c", done=True),
        ]

        self.check(max_minutes="0")

        self.assertEqual(self.cancelled(), ["a", "b"])
        self.assertIn("Kill switch: running longer than 0:00:00; cancelling", self.log.getvalue())

    def test_kill_switch_with_a_limit(self):
        self.cloud.executions = [
            execution("train", "long", minutes_ago=45),
            execution("train", "short", minutes_ago=10),
        ]

        self.check(max_minutes="30")

        self.assertEqual(self.cancelled(), ["long"])

    def test_reads_every_page(self):
        self.cloud.executions = [
            execution(f"job{j}", f"run{j}-{i}", minutes_ago=9 * 60) for j in range(5) for i in range(3)
        ]

        result = self.check()

        self.assertEqual(result["checked"], 15)
        self.assertEqual(len(self.cloud.cancelled), 15)


class BudgetTest(WatchdogTestCase):
    def setUp(self):
        super().setUp()
        self.cloud.executions = [
            execution("train", "new", minutes_ago=1),
            execution("eval", "done", done=True),
        ]

    def test_under_budget(self):
        self.cloud.cost = 299.99

        result = self.check()

        self.assertEqual(result["month_cost"], 299.99)
        self.assertEqual(self.cancelled(), [])
        self.assertIn("This month's cost so far: $299.99 of $300", self.log.getvalue())

    def test_at_budget_cancels_everything_running(self):
        self.cloud.cost = 300.0

        self.check()

        self.assertEqual(self.cancelled(), ["new"])
        self.assertIn("OVER BUDGET", self.log.getvalue())
        self.assertIn("Over budget; cancelling", self.log.getvalue())
        self.assertNotIn("Running longer than", self.log.getvalue())

    def test_queries_this_project_this_month(self):
        self.check()

        (query,) = self.cloud.queries
        self.assertIn(f"`{TABLE}`", query["query"])
        self.assertIn("America/Los_Angeles", query["query"])
        self.assertIn("project.id = @project", query["query"])
        self.assertFalse(query["useLegacySql"])
        (param,) = query["queryParameters"]
        self.assertEqual(param["parameterValue"]["value"], PROJECT)

    def test_no_costs_yet_this_month(self):
        self.cloud.cost = None

        self.assertEqual(self.check()["month_cost"], 0.0)

    def test_billing_export_not_there_yet(self):
        self.cloud.billing = "missing"
        self.cloud.executions.append(execution("train", "weekend", minutes_ago=9 * 60))

        result = self.check()

        self.assertIsNone(result["month_cost"])
        self.assertIn("No billing export data yet", self.log.getvalue())
        self.assertEqual(self.cancelled(), ["weekend"])

    def test_a_broken_cost_check_still_runs_the_other_checks(self):
        for billing in ("error", "slow"):
            with self.subTest(billing):
                self.cloud.billing = billing
                self.cloud.cancelled = []
                self.cloud.executions = [execution("train", "weekend", minutes_ago=9 * 60)]

                result = self.check()

                self.assertIsNone(result["month_cost"])
                self.assertIn("Couldn't read this month's cost", self.log.getvalue())
                self.assertEqual(self.cancelled(), ["weekend"])


ALICE = "wh-alice@test-project.iam.gserviceaccount.com"
BOB = "wh-bob@test-project.iam.gserviceaccount.com"
CAROL = "wh-carol@test-project.iam.gserviceaccount.com"


class GpuCapTest(WatchdogTestCase):
    def test_a_person_over_5_gpus_loses_their_newest_executions(self):
        self.cloud.executions = [
            execution("train", f"run{i}", gpus=1, minutes_ago=60 - i) for i in range(7)
        ]

        self.check()

        self.assertEqual(self.cancelled(), ["run5", "run6"])
        self.assertIn(
            "GPU cap: this would put wh-alice@test-project.iam.gserviceaccount.com over 5 GPUs; cancelling",
            self.log.getvalue(),
        )

    def test_one_oversized_launch_is_cancelled_and_older_work_kept(self):
        self.cloud.executions = [
            execution("train", "small", gpus=1, tasks=2, minutes_ago=30),
            execution("sweep", "huge", gpus=1, tasks=50, parallelism=0, minutes_ago=1),
        ]

        self.check()

        self.assertEqual(self.cancelled(), ["huge"])

    def test_parallelism_limits_how_many_gpus_an_execution_holds(self):
        self.cloud.executions = [
            # 50 tasks, but only 5 at a time: exactly at the cap.
            execution("sweep", "throttled", gpus=1, tasks=50, parallelism=5),
        ]

        self.check()

        self.assertEqual(self.cancelled(), [])

    def test_counts_gpus_per_task(self):
        self.cloud.executions = [
            execution("train", "two-by-three", gpus=2, tasks=3, parallelism=3),
        ]

        self.check()

        self.assertEqual(self.cancelled(), ["two-by-three"])

    def test_finished_tasks_give_their_gpus_back(self):
        self.cloud.executions = [
            # 8 tasks, 6 done: 2 GPUs still in use.
            execution("sweep", "mostly-done", gpus=1, tasks=8, succeededCount=4, failedCount=1, cancelledCount=1),
            execution("train", "new", gpus=1, tasks=3, minutes_ago=1),
            execution("eval", "done", gpus=1, tasks=20, done=True),
        ]

        self.check()

        self.assertEqual(self.cancelled(), [])

    def test_cpu_only_jobs_never_count(self):
        self.cloud.executions = [
            execution("etl", "big-cpu", tasks=100, parallelism=0),
            execution("train", "gpu", gpus=1, tasks=5, minutes_ago=1),
        ]

        self.check()

        self.assertEqual(self.cancelled(), [])

    def test_each_person_has_their_own_cap(self):
        self.cloud.executions = [
            execution("a", "alice", creator=ALICE, gpus=1, tasks=5, minutes_ago=3),
            execution("b", "bob", creator=BOB, gpus=1, tasks=5, minutes_ago=2),
        ]

        self.check()

        self.assertEqual(self.cancelled(), [])

    def test_the_project_holds_10_gpus_at_most(self):
        self.cloud.executions = [
            execution("a", "alice", creator=ALICE, gpus=1, tasks=4, minutes_ago=3),
            execution("b", "bob", creator=BOB, gpus=1, tasks=4, minutes_ago=2),
            execution("c", "carol-big", creator=CAROL, gpus=1, tasks=3, minutes_ago=1),
            execution("c", "carol-small", creator=CAROL, gpus=1, tasks=2, minutes_ago=0.5),
        ]

        self.check()

        # Carol's 3 would make 11; her later 2 still fit.
        self.assertEqual(self.cancelled(), ["carol-big"])
        self.assertIn("this would put the project over 10 GPUs", self.log.getvalue())

    def test_an_execution_cancelled_for_running_long_frees_its_gpus(self):
        self.cloud.executions = [
            execution("train", "weekend", gpus=1, tasks=5, minutes_ago=9 * 60),
            execution("train", "today", gpus=1, tasks=5, minutes_ago=10),
        ]

        self.check()

        # Cancelled once, for running long; today's fits once it's gone.
        self.assertEqual(self.cancelled(), ["weekend"])


if __name__ == "__main__":
    unittest.main()
