// The list of runs, and a run's page: its episodes, and the chosen one.

import { api, Me, Run, SignInNeeded } from "./api";
import { duration, h, link, navigate, when } from "./dom";
import { defaultPrefs, episodeView, EpisodeView } from "./episode";

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

/** A run's page: its episodes down the left, the chosen one on the right. */
export interface RunView {
  element: HTMLElement;
  /** Show this episode, or the first one with data if `index` is null. */
  select: (index: number | null) => Promise<void>;
  stop: () => void;
}

const REFRESH_MS = 15_000;

export async function runView(id: string): Promise<RunView> {
  let run = await api.run(id);
  const prefs = defaultPrefs();
  let current: EpisodeView | null = null;
  let selected: number | null = null;
  let request = 0;

  const facts = h("details", { class: "facts-box" });
  const list = h("ol", { class: "playlist" });
  const pane = h("div", { class: "episode-pane" });

  function drawFacts() {
    const rows: [string, Node | string][] = [
      ["Launched by", run.launched_by],
      ["When", when(run.created_at)],
      ["Command", h("code", null, run.command.join(" "))],
    ];
    if (run.sim) rows.push(["Backend", run.sim.backend]);
    if (run.git) rows.push(["Code", `${run.git.branch} at ${run.git.commit.slice(0, 7)}${run.git.dirty ? ", with uncommitted changes" : ""}`]);
    if (run.gpu) rows.push(["GPU", "yes"]);
    facts.replaceChildren(
      h("summary", null, "Details"),
      h("dl", { class: "facts" }, ...rows.flatMap(([k, v]) => [h("dt", null, k), h("dd", null, v)])),
    );
  }

  function drawList() {
    list.replaceChildren(
      ...run.episodes_detail.map((ep) =>
        h("li", null,
          link(`/runs/${run.id}/episodes/${ep.index}`,
            h("span", { class: "ep-name" }, `Episode ${ep.index}`),
            h("span", { class: `status ${ep.status}` }, ep.status),
            h("span", { class: "ep-meta" },
              h("span", { class: "mono" }, `seed ${ep.seed}`),
              duration(ep.started_at, ep.finished_at) && ` · ${duration(ep.started_at, ep.finished_at)}`),
          ),
        ),
      ),
    );
    markSelected();
  }

  function markSelected() {
    list.querySelectorAll("li").forEach((li, i) => {
      const on = run.episodes_detail[i]?.index === selected;
      li.classList.toggle("selected", on);
      if (on) li.querySelector("a")?.setAttribute("aria-current", "page");
      else li.querySelector("a")?.removeAttribute("aria-current");
    });
  }

  async function select(index: number | null) {
    if (index === null) {
      const first = run.episodes_detail.find((ep) => ep.sim) ?? run.episodes_detail[0];
      if (!first) {
        pane.replaceChildren(h("p", { class: "muted" }, "This run has no episodes."));
        return;
      }
      index = first.index;
      history.replaceState(null, "", `/runs/${run.id}/episodes/${index}`);
    }
    if (index === selected && current) return;
    current?.stop();
    current = null;
    selected = index;
    markSelected();
    list.querySelector("li.selected")?.scrollIntoView({ block: "nearest" });
    const mine = ++request;
    pane.replaceChildren(h("p", { class: "muted" }, `Loading episode ${index}...`));
    try {
      const view = await episodeView(run.id, index, prefs);
      if (mine !== request) return view.stop(); // another was chosen meanwhile
      current = view;
      pane.replaceChildren(view.element);
    } catch (err) {
      if (err instanceof SignInNeeded) throw err;
      if (mine === request) pane.replaceChildren(h("p", { class: "error" }, (err as Error).message));
    }
  }

  // ↑ and ↓ move through the episodes.
  const keys = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "SELECT" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    const at = run.episodes_detail.findIndex((ep) => ep.index === selected);
    const next = run.episodes_detail[at + (e.key === "ArrowDown" ? 1 : -1)];
    if (next) navigate(`/runs/${run.id}/episodes/${next.index}`);
  };
  document.addEventListener("keydown", keys);

  // While episodes are still going, keep their statuses fresh.
  const timer = setInterval(async () => {
    if (!run.episodes_detail.some((ep) => ep.status === "pending" || ep.status === "running")) return;
    try {
      run = await api.run(id);
      drawList();
    } catch {
      // Try again next time.
    }
  }, REFRESH_MS);

  drawFacts();
  drawList();
  const what = run.sim ? `${run.sim.scenario.split(":").pop()} in ${run.sim.scene}` : run.command.join(" ");
  const element = h("section", { class: "run" },
    h("aside", { class: "sidebar" },
      h("p", null, link("/", "← Runs")),
      h("h1", { class: "run-id mono" }, run.id),
      h("p", { class: "muted" }, what),
      facts,
      h("h2", null, `Episodes (${run.episodes_detail.length})`),
      list,
      h("p", { class: "muted small" }, "↑ ↓ episodes · ← → frames · space plays"),
    ),
    pane,
  );
  return {
    element,
    select,
    stop: () => {
      clearInterval(timer);
      document.removeEventListener("keydown", keys);
      current?.stop();
    },
  };
}
