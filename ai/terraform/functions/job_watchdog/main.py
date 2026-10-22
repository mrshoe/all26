"""Cancel Cloud Run job executions that have run too long.

Cloud Scheduler calls this every 15 minutes. Any execution in the region that
has been running for more than MAX_RUNTIME_HOURS is cancelled. This catches a
job left running over a weekend, including one whose timeout was overridden
at launch.

Manual kill switch: a mentor can call it with ?max_minutes=0 to cancel every
running execution now. Only callers with run.invoker (mentors, and the
scheduler's service account) can reach it.
"""

import datetime
import json
import os
import urllib.request

import functions_framework

PROJECT = os.environ["PROJECT"]
REGION = os.environ["REGION"]
MAX_RUNTIME_HOURS = float(os.environ["MAX_RUNTIME_HOURS"])
RUN_API = "https://run.googleapis.com/v2"
METADATA_TOKEN = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/token"
)


def access_token() -> str:
    req = urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["access_token"]


def call(token: str, method: str, url: str) -> dict:
    req = urllib.request.Request(
        url,
        data=b"" if method == "POST" else None,
        headers={"Authorization": f"Bearer {token}"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def list_all(token: str, url: str, field: str) -> list[dict]:
    items, page = [], ""
    while True:
        reply = call(token, "GET", f"{url}?pageSize=100&pageToken={page}")
        items += reply.get(field, [])
        page = reply.get("nextPageToken")
        if not page:
            return items


def parse_time(stamp: str) -> datetime.datetime:
    # Cloud Run uses RFC 3339 with up to nanoseconds; seconds are plenty.
    return datetime.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )


def overdue_executions(executions: list[dict], limit: datetime.timedelta, now) -> list[str]:
    """Names of executions still running after `limit`."""
    overdue = []
    for execution in executions:
        if execution.get("completionTime"):
            continue
        started = execution.get("startTime") or execution.get("createTime")
        if started and now - parse_time(started) > limit:
            overdue.append(execution["name"])
    return overdue


@functions_framework.http
def check_jobs(request):
    max_minutes = request.args.get("max_minutes")
    if max_minutes is not None:
        limit = datetime.timedelta(minutes=float(max_minutes))
    else:
        limit = datetime.timedelta(hours=MAX_RUNTIME_HOURS)
    now = datetime.datetime.now(datetime.timezone.utc)

    token = access_token()
    executions = []
    for job in list_all(token, f"{RUN_API}/projects/{PROJECT}/locations/{REGION}/jobs", "jobs"):
        executions += list_all(token, f"{RUN_API}/{job['name']}/executions", "executions")

    cancelled = overdue_executions(executions, limit, now)
    for name in cancelled:
        print(f"Running longer than {limit}; cancelling {name}")
        call(token, "POST", f"{RUN_API}/{name}:cancel")
    print(f"Checked {len(executions)} executions; cancelled {len(cancelled)}.")
    return {"checked": len(executions), "cancelled": cancelled}
