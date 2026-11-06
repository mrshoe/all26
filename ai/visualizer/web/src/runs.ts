// The list of runs, and a run's page: its episodes, and the chosen one.

import { api, Me, Run, SignInNeeded } from "./api";
import { duration, h, icon, link, navigate, when } from "./dom";
import { defaultPrefs, episodeView, EpisodeView } from "./episode";

const FILTERS = "wildhat.filters";

function what(run: Run): (Node | string)[] {
  if (run.sim) return [`${run.sim.scenario.split(":").pop()} in ${run.sim.scene}`];
  if (run.photos) return [h("span", { class: "tag", title: "Labeled photos, not simulated" }, "photos"), " ", run.photos.name];
  const command = (run.command ?? []).join(" ");
  if (run.training) return [trainingBadge(run.training), " ", command];
  return [command];
}

/** A job that ran a training run: a badge linking to it. */
function trainingBadge(id: string): HTMLElement {
  return link(`/training/${id}`, h("span", { class: "tag", title: `Training run ${id}` }, "training"));
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
          h("td", { class: "what" }, ...what(run)),
          h("td", { class: "num" }, run.episodes),
          h("td", null, statuses(run.status_counts)),
        ),
      ),
    );
    if (!runs.length) {
      body.append(h("tr", null, h("td", { colspan: 6, class: "muted" }, "No runs match.")));
    }
  }

  // A chip that's lit while its filter is on, like the Hattracker's.
  const toggle = (label: string, key: "mine" | "sim") => {
    const chip = h("button", {
      type: "button",
      class: "chip",
      "aria-pressed": String(filters[key]),
      onclick: () => {
        filters[key] = !filters[key];
        chip.setAttribute("aria-pressed", String(filters[key]));
        refresh();
      },
    }, label);
    return chip;
  };

  await refresh();
  return h("section", null,
    h("div", { class: "toolbar" }, h("h1", null, "Runs"),
      h("div", { class: "chips" }, toggle("Only mine", "mine"), toggle("Only simulations", "sim")), count),
    h("div", { class: "table-wrap" },
      h("table", { class: "runs" },
        h("thead", null, h("tr", null, ...["Run", "Who", "When", "What", "Episodes", "Status"].map((t) => h("th", null, t)))),
        body,
      ),
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
      [run.photos ? "Imported by" : "Launched by", run.launched_by],
      ["When", when(run.created_at)],
    ];
    if (run.command) rows.push(["Command", h("code", null, run.command.join(" "))]);
    if (run.photos) {
      const p = run.photos;
      rows.push(["Photos", p.url ? h("a", { href: p.url, target: "_blank", rel: "noopener" }, p.name) : p.name]);
      if (p.credit) rows.push(["Credit", p.credit]);
      if (p.license) rows.push(["License", p.license]);
      rows.push(["Labels", p.labels.join(", ")]);
    }
    if (run.sim) rows.push(["Backend", run.sim.backend]);
    if (run.git) rows.push(["Code", `${run.git.branch} at ${run.git.commit.slice(0, 7)}${run.git.dirty ? ", with uncommitted changes" : ""}`]);
    if (run.gpu) rows.push(["GPU", "yes"]);
    if (run.training) rows.push(["Training run", link(`/training/${run.training}`, h("span", { class: "mono" }, run.training))]);
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
              ep.photos
                ? h("span", null, `${ep.photos.split} · ${ep.photos.frames} photos`)
                : h("span", { class: "mono" }, `seed ${ep.seed}`),
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
      const first = run.episodes_detail.find((ep) => ep.sim || ep.photos) ?? run.episodes_detail[0];
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
  const element = h("section", { class: "run" },
    h("aside", { class: "sidebar" },
      h("p", null, link("/", icon("back"), "Runs")),
      h("h1", { class: "run-id mono" }, run.id),
      h("p", { class: "muted" }, ...what(run)),
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
