// The list of runs, and one run's episodes.

import { api, Me, Run } from "./api";
import { duration, h, link, when } from "./dom";

const FILTERS = "wildhat.filters";

function what(run: Run): string {
  return run.sim ? `${run.sim.scenario.split(":").pop()} in ${run.sim.scene}` : run.command.join(" ");
}

function statuses(counts: Record<string, number>): HTMLElement {
  const order = ["running", "pending", "succeeded", "failed", "stopped"];
  return h("span", { class: "statuses" },
    ...Object.entries(counts)
      .sort(([a], [b]) => order.indexOf(a) - order.indexOf(b))
      .map(([status, n]) => h("span", { class: `status ${status}` }, `${n} ${status}`)),
  );
}

function loadFilters(): { mine: boolean; sim: boolean } {
  try {
    return { mine: false, sim: true, ...JSON.parse(localStorage.getItem(FILTERS) ?? "{}") };
  } catch {
    return { mine: false, sim: true };
  }
}

export async function runsPage(me: Me): Promise<HTMLElement> {
  const filters = loadFilters();
  const body = h("tbody");
  const count = h("span", { class: "muted" });

  async function refresh() {
    try {
      localStorage.setItem(FILTERS, JSON.stringify(filters));
    } catch {
      // Not remembered, that's all.
    }
    const runs = await api.runs({ user: filters.mine ? me.handle : undefined, sim: filters.sim });
    count.textContent = `${runs.length} run${runs.length === 1 ? "" : "s"}`;
    body.replaceChildren(
      ...runs.map((run) =>
        h("tr", null,
          h("td", null, link(`/runs/${run.id}`, run.id)),
          h("td", null, run.launched_by),
          h("td", null, when(run.created_at)),
          h("td", { class: "what" }, what(run)),
          h("td", { class: "num" }, run.episodes),
          h("td", null, statuses(run.status_counts)),
        ),
      ),
    );
    if (!runs.length) {
      body.append(h("tr", null, h("td", { colspan: 6, class: "muted" }, "No runs match.")));
    }
  }

  const toggle = (label: string, key: "mine" | "sim") =>
    h("label", null,
      h("input", {
        type: "checkbox",
        checked: filters[key],
        onchange: (e: Event) => {
          filters[key] = (e.target as HTMLInputElement).checked;
          refresh();
        },
      }),
      " ", label);

  await refresh();
  return h("section", null,
    h("div", { class: "toolbar" }, h("h1", null, "Runs"), toggle("Only mine", "mine"), toggle("Only simulations", "sim"), count),
    h("table", { class: "runs" },
      h("thead", null, h("tr", null, ...["Run", "Who", "When", "What", "Episodes", "Status"].map((t) => h("th", null, t)))),
      body,
    ),
  );
}

export async function runPage(id: string): Promise<HTMLElement> {
  const run = await api.run(id);
  const facts: [string, Node | string][] = [
    ["Launched by", run.launched_by],
    ["When", when(run.created_at)],
    ["Command", h("code", null, run.command.join(" "))],
  ];
  if (run.sim) facts.push(["Scenario", `${run.sim.scenario} in ${run.sim.scene} (${run.sim.backend})`]);
  if (run.git) facts.push(["Code", `${run.git.branch} at ${run.git.commit.slice(0, 7)}${run.git.dirty ? ", with uncommitted changes" : ""}`]);
  if (run.gpu) facts.push(["GPU", "yes"]);

  return h("section", null,
    h("p", null, link("/", "← Runs")),
    h("h1", null, run.id),
    h("dl", { class: "facts" }, ...facts.flatMap(([k, v]) => [h("dt", null, k), h("dd", null, v)])),
    h("table", { class: "runs" },
      h("thead", null, h("tr", null, ...["Episode", "Status", "Seed", "Time", "Frames"].map((t) => h("th", null, t)))),
      h("tbody", null,
        ...run.episodes_detail.map((ep) =>
          h("tr", null,
            h("td", null, ep.sim ? link(`/runs/${run.id}/episodes/${ep.index}`, `Episode ${ep.index}`) : `Episode ${ep.index}`),
            h("td", null, h("span", { class: `status ${ep.status}` }, ep.status)),
            h("td", { class: "mono" }, String(ep.seed)),
            h("td", null, duration(ep.started_at, ep.finished_at)),
            h("td", { class: "num" }, ep.sim?.frames ?? ""),
          ),
        ),
      ),
    ),
  );
}
