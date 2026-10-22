"""Cancel Cloud Run job executions that run too long or overspend.

Cloud Scheduler calls this every 15 minutes. It cancels:

- any execution in the region running for more than MAX_RUNTIME_HOURS. This
  catches a job left running over a weekend, including one whose timeout was
  overridden at launch.
- every running execution, once this month's cost in the billing export has
  reached BUDGET_USD. Cost is counted before credits, so trial or nonprofit
  credits don't hide spending. The export lags real use by hours to a day,
  so this is a backstop, not a precise cap. It doesn't stop new launches;
  each later run cancels again until the budget is raised or the month ends.

Manual kill switch: a mentor can call it with ?max_minutes=0 to cancel every
running execution now. Only callers with run.invoker (mentors, and the
scheduler's service account) can reach it.
"""

import datetime
import json
import os
import urllib.error
import urllib.request

import functions_framework

PROJECT = os.environ["PROJECT"]
REGION = os.environ["REGION"]
MAX_RUNTIME_HOURS = float(os.environ["MAX_RUNTIME_HOURS"])
BUDGET_USD = float(os.environ["BUDGET_USD"])
BILLING_TABLE = os.environ["BILLING_TABLE"]
RUN_API = "https://run.googleapis.com/v2"
BIGQUERY_API = "https://bigquery.googleapis.com/bigquery/v2"
METADATA_TOKEN = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/token"
)


def access_token() -> str:
    req = urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["access_token"]


def call(token: str, method: str, url: str, payload: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if method == "POST":
        data = json.dumps(payload or {}).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
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


# Billing months follow the billing account's time zone, US Pacific.
MONTH_COST_SQL = """
SELECT SUM(cost)
FROM `{table}`
WHERE invoice.month = FORMAT_DATE('%Y%m', CURRENT_DATE('America/Los_Angeles'))
  AND project.id = @project
"""


def month_cost(token: str) -> float | None:
    """This month's cost before credits, or None if the export has no table yet."""
    try:
        reply = call(
            token,
            "POST",
            f"{BIGQUERY_API}/projects/{PROJECT}/queries",
            {
                "query": MONTH_COST_SQL.format(table=BILLING_TABLE),
                "useLegacySql": False,
                "timeoutMs": 60000,
                "parameterMode": "NAMED",
                "queryParameters": [
                    {
                        "name": "project",
                        "parameterType": {"type": "STRING"},
                        "parameterValue": {"value": PROJECT},
                    }
                ],
            },
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    if not reply.get("jobComplete"):
        raise RuntimeError("billing query didn't finish in time")
    value = reply["rows"][0]["f"][0]["v"]
    return float(value) if value is not None else 0.0


def runtime_limit(max_minutes: str | None, cost: float | None) -> datetime.timedelta:
    """How long an execution may run before it's cancelled."""
    if max_minutes is not None:
        return datetime.timedelta(minutes=float(max_minutes))
    if cost is not None and cost >= BUDGET_USD:
        return datetime.timedelta(0)
    return datetime.timedelta(hours=MAX_RUNTIME_HOURS)


@functions_framework.http
def check_jobs(request):
    token = access_token()
    # A broken cost check must not stop the runtime check.
    cost = None
    try:
        cost = month_cost(token)
        if cost is None:
            print("No billing export data yet.")
    except Exception as e:
        print(f"Couldn't read this month's cost: {e}")
    if cost is not None:
        print(f"This month's cost so far: ${cost:.2f} of ${BUDGET_USD:.0f}")
        if cost >= BUDGET_USD:
            print("OVER BUDGET: cancelling every running execution.")
    limit = runtime_limit(request.args.get("max_minutes"), cost)
    now = datetime.datetime.now(datetime.timezone.utc)

    executions = []
    for job in list_all(token, f"{RUN_API}/projects/{PROJECT}/locations/{REGION}/jobs", "jobs"):
        executions += list_all(token, f"{RUN_API}/{job['name']}/executions", "executions")

    cancelled = overdue_executions(executions, limit, now)
    for name in cancelled:
        print(f"Running longer than {limit}; cancelling {name}")
        call(token, "POST", f"{RUN_API}/{name}:cancel")
    print(f"Checked {len(executions)} executions; cancelled {len(cancelled)}.")
    return {"month_cost": cost, "checked": len(executions), "cancelled": cancelled}
