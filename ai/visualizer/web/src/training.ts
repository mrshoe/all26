// Training runs (ai/src/shared/training.py records them):
//
//   /training                     the list; tick runs to compare them
//   /training/<id>                one run: its facts, a plot per metric
//                                 (live while it runs), and its checkpoints
//   /training/compare?runs=a,b    those runs' metrics, overlaid
//
// Metrics are plotted against step. A metric logged only now and then (a
// validation score, say) is drawn through the steps it has.

import uPlot from "uplot";
import { api, Checkpoint, Me, MetricRow, TrainingDetail, Where } from "./api";
import { duration, h, icon, link, navigate, when } from "./dom";

// The same as shared/training.py's: a running run says it's alive every
// minute, and is taken to have stopped after 3 minutes of silence.
const STALE_MS = 180_000;
const REFRESH_MS = 30_000;
const FILTERS = "wildhat.trainingFilters";

// As in the episode plots (episode.ts): none is orange.
const COLORS = ["#4da3ff", "#46c98b", "#b98cff", "#ffb340", "#ff7ab0", "#e8e4e1"];
const GRID = "rgba(243, 242, 242, 0.08)";
const AXIS = "#a8a09d";

/** Its status, or "stopped?" for one still marked running that's gone quiet. */
export function statusOf(run: { status: string; heartbeat_at?: string }): string {
  if (run.status === "running" && run.heartbeat_at && Date.now() - new Date(run.heartbeat_at).getTime() > STALE_MS) {
    return "stopped?";
  }
  return run.status;
}

function statusPill(run: { status: string; heartbeat_at?: string }): HTMLElement {
  const status = statusOf(run);
  const kind = status === "finished" ? "succeeded" : status === "stopped?" ? "stopped" : status;
  return h("span", {
    class: `status ${kind}`,
    title: status === "stopped?" ? "Still marked running, but not heard from for 3 minutes: its computer closed or went offline, say." : undefined,
  }, status);
}

function whereText(where?: Where): string {
  if (!where) return "";
  return where.kind === "cloud" ? `cloud run ${where.run}, episode ${where.episode ?? 0}` : `laptop ${where.host ?? ""}`;
}

function metricsText(metrics?: Record<string, number> | null): string {
  return Object.entries(metrics ?? {}).map(([k, v]) => `${k} ${format(v)}`).join("  ");
}

function format(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 0.001 || a >= 1e5) ? v.toExponential(2) : String(Number(v.toPrecision(4)));
}

function bytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return n >= 1e3 ? `${Math.round(n / 1e3)} KB` : `${n} B`;
}

/** A link to a training run, or to its run page for "<run>/<step>". */
function runLink(checkpoint: string): HTMLAnchorElement {
  const run = checkpoint.split("/")[0];
  return link(`/training/${run}`, h("span", { class: "mono" }, checkpoint));
}

// ---------------------------------------------------------------------------
// Plots: one per metric, each run a series
// ---------------------------------------------------------------------------

interface Series {
  label: string;
  rows: MetricRow[];
}

/** Every metric the runs logged, in the order first seen. */
function metricNames(series: Series[]): string[] {
  const names = new Set<string>();
  for (const s of series) for (const row of s.rows) for (const k of Object.keys(row)) if (k !== "step" && k !== "time") names.add(k);
  return [...names];
}

class MetricPlots {
  element = h("div", { class: "metric-grid" });
  private plots: uPlot[] = [];
  private observer = new ResizeObserver(() => this.resize());

  constructor(private emptyText: string) {
    this.observer.observe(this.element);
  }

  draw(series: Series[]): void {
    this.destroy();
    const names = metricNames(series);
    if (!names.length) {
      this.element.replaceChildren(h("p", { class: "muted" }, this.emptyText));
      return;
    }
    this.element.replaceChildren();
    for (const name of names) {
      // Every run's steps, so the series line up; each run's values where it has them.
      const steps = [...new Set(series.flatMap((s) => s.rows.filter((r) => name in r).map((r) => r.step)))].sort((a, b) => a - b);
      const data: (number | null)[][] = [steps];
      for (const s of series) {
        const at = new Map<number, number>();
        for (const r of s.rows) if (name in r) at.set(r.step, r[name]);
        data.push(steps.map((step) => at.get(step) ?? null));
      }
      const holder = h("div", { class: "plot" });
      const card = h("div", { class: "plot-card" }, h("h3", { class: "metric-name mono" }, name), holder);
      this.element.append(card);
      const plot = new uPlot({
        width: this.width(holder),
        height: 200,
        scales: { x: { time: false } },
        axes: [
          { label: "step", stroke: AXIS, grid: { stroke: GRID }, ticks: { stroke: GRID } },
          { stroke: AXIS, grid: { stroke: GRID }, ticks: { stroke: GRID }, size: 60 },
        ],
        series: [
          { label: "step", value: (_u: uPlot, v: number | null) => (v == null ? "–" : String(v)) },
          ...series.map((s, i) => ({
            // One run: the legend names the metric; several: each run.
            label: series.length === 1 ? name : s.label,
            stroke: COLORS[i % COLORS.length],
            width: 2,
            // Metrics logged now and then are drawn through the steps they have.
            spanGaps: true,
            points: { show: steps.length < 40 },
            value: (_u: uPlot, v: number | null) => (v == null ? "–" : format(v)),
          })),
        ],
        cursor: { drag: { x: true, y: false } },
      }, data as uPlot.AlignedData, holder);
      this.plots.push(plot);
    }
  }

  /** The card's inner width: the plot fits its card, never the reverse. */
  private width(holder: HTMLElement): number {
    const card = holder.parentElement as HTMLElement;
    return Math.max(200, (card.clientWidth || 440) - 20);
  }

  private resize(): void {
    for (const plot of this.plots) plot.setSize({ width: this.width(plot.root.parentElement as HTMLElement), height: 200 });
  }

  destroy(): void {
    this.plots.forEach((p) => p.destroy());
    this.plots = [];
  }

  stop(): void {
    this.destroy();
    this.observer.disconnect();
  }
}

// ---------------------------------------------------------------------------
// The list
// ---------------------------------------------------------------------------

function loadFilters(): { mine: boolean; model: string } {
  try {
    return { mine: false, model: "", ...JSON.parse(localStorage.getItem(FILTERS) ?? "{}") };
  } catch {
    return { mine: false, model: "" };
  }
}

export async function trainingPage(me: Me): Promise<HTMLElement> {
  const filters = loadFilters();
  const chosen = new Set<string>();
  const body = h("tbody");
  const count = h("span", { class: "muted" });
  const models = h("select", { "aria-label": "Model" });
  const compare = h("button", {
    class: "btn btn--primary btn--sm",
    disabled: true,
    onclick: () => navigate(`/training/compare?runs=${[...chosen].join(",")}`),
  }, "Compare");
  const known = new Set<string>();

  function updateCompare() {
    compare.disabled = chosen.size < 2;
    compare.textContent = chosen.size ? `Compare ${chosen.size}` : "Compare";
  }

  function drawModels() {
    if (filters.model) known.add(filters.model);
    models.replaceChildren(
      h("option", { value: "" }, "All models"),
      ...[...known].sort().map((m) => h("option", { value: m, selected: m === filters.model }, m)),
    );
  }

  async function refresh() {
    try {
      localStorage.setItem(FILTERS, JSON.stringify(filters));
    } catch {
      // Not remembered, that's all.
    }
    const runs = await api.trainingRuns({ model: filters.model || undefined, user: filters.mine ? me.handle : undefined });
    runs.forEach((r) => known.add(r.model));
    drawModels();
    count.textContent = `${runs.length} training run${runs.length === 1 ? "" : "s"}`;
    body.replaceChildren(...runs.map((run) => {
      const box = h("input", {
        type: "checkbox",
        "aria-label": `Compare ${run.id}`,
        checked: chosen.has(run.id),
        onchange: (e: Event) => {
          if ((e.target as HTMLInputElement).checked) chosen.add(run.id);
          else chosen.delete(run.id);
          updateCompare();
        },
      });
      return h("tr", null,
        h("td", null, box),
        h("td", null, link(`/training/${run.id}`, run.id)),
        h("td", null, run.model),
        h("td", null, run.trained_by),
        h("td", null, when(run.created_at)),
        h("td", null, statusPill(run), run.resumed ? h("span", { class: "muted small" }, ` resumed ${run.resumed}×`) : null),
        h("td", { class: "num" }, run.step ?? ""),
        h("td", { class: "mono small what" }, metricsText(run.latest)),
        h("td", { class: "small" }, whereText(run.where)),
      );
    }));
    if (!runs.length) body.append(h("tr", null, h("td", { colspan: 9, class: "muted" }, "No training runs match.")));
  }

  models.addEventListener("change", () => {
    filters.model = models.value;
    refresh();
  });
  const mine = h("button", {
    type: "button",
    class: "chip",
    "aria-pressed": String(filters.mine),
    onclick: () => {
      filters.mine = !filters.mine;
      mine.setAttribute("aria-pressed", String(filters.mine));
      refresh();
    },
  }, "Only mine");

  await refresh();
  return h("section", null,
    h("div", { class: "toolbar" }, h("h1", null, "Training"),
      h("div", { class: "chips" }, mine), models, count, h("span", { class: "header__spacer" }), compare),
    h("div", { class: "table-wrap" },
      h("table", { class: "runs" },
        h("thead", null, h("tr", null,
          ...["", "Run", "Model", "Who", "When", "Status", "Step", "Latest", "Where"].map((t) => h("th", null, t)))),
        body,
      ),
    ),
    h("p", { class: "muted small" }, "Tick two or more runs to compare their metrics."),
  );
}

// ---------------------------------------------------------------------------
// One run
// ---------------------------------------------------------------------------

export interface TrainingView {
  element: HTMLElement;
  stop: () => void;
}

function checkpointsTable(id: string, checkpoints: Checkpoint[], best: number | null): HTMLElement {
  if (!checkpoints.length) return h("p", { class: "muted" }, "No checkpoints yet.");
  return h("div", { class: "table-wrap" },
    h("table", null,
      h("thead", null, h("tr", null, ...["Checkpoint", "Size", "Saved", "Kind", "Metrics", "Tags"].map((t) => h("th", null, t)))),
      h("tbody", null, ...checkpoints.map((c) =>
        h("tr", { class: c.pruned ? "pruned" : undefined },
          h("td", { class: "mono" }, `${id}/${c.step}`),
          h("td", { class: "num" }, bytes(c.size)),
          h("td", null, when(c.saved_at)),
          // The run's best, not every checkpoint that was best when saved.
          h("td", null, [c.step === best && "best", c.resumable && "resumable", c.pruned && "pruned"].filter(Boolean).join(", ")),
          h("td", { class: "mono small" }, metricsText(c.metrics)),
          h("td", null, ...(c.tags ?? []).map((t) => h("span", { class: "tag" }, t))),
        ))),
    ),
  );
}

function facts(run: TrainingDetail): HTMLElement {
  const rows: [string, Node | string | (Node | string)[]][] = [
    ["Model", run.model],
    ["By", run.trained_by],
    ["Started", when(run.created_at)],
  ];
  if (run.finished_at) rows.push(["Ended", `${when(run.finished_at)}, after ${duration(run.created_at, run.finished_at)}`]);
  if (run.error) rows.push(["Error", h("code", null, run.error)]);
  run.segments.forEach((s, i) => {
    const git = s.git ? `${s.git.branch} at ${s.git.commit.slice(0, 7)}${s.git.dirty ? ", with uncommitted changes" : ""}` : "";
    rows.push([i === 0 ? "Code" : `Resumed at step ${s.resumed_at_step}`, `${git} · ${whereText(s.where ?? run.where)}`]);
  });
  const samples = Object.entries(run.dataset.samples ?? {}).map(([k, v]) => `${k} ${v}`).join(", ");
  const episodes = Object.entries(run.dataset.episodes ?? {}).map(([k, v]) => `${k} ${v}`).join(", ");
  rows.push(["Dataset", [h("code", null, run.dataset.path), h("span", { class: "muted" }, ` · samples: ${samples} · episodes: ${episodes}`)]]);
  for (const [name, p] of Object.entries(run.parents ?? {})) {
    const asked = p.ref !== p.checkpoint ? ` (asked as ${p.ref})` : "";
    rows.push([`Started from: ${name}`, [runLink(p.checkpoint), ` ${p.model}${asked}`]]);
  }
  if (run.children.length) {
    rows.push(["Used by", run.children.flatMap((c, i) => [i ? ", " : "", link(`/training/${c.id}`, h("span", { class: "mono" }, c.id)), ` ${c.model}`])]);
  }
  if (run.best) rows.push(["Best", `step ${run.best.step}   ${metricsText(run.best.metrics)}`]);
  return h("dl", { class: "facts" }, ...rows.flatMap(([k, v]) => [h("dt", null, k), h("dd", null, ...(Array.isArray(v) ? v : [v]))]));
}

export async function trainingRunView(id: string): Promise<TrainingView> {
  let [run, metrics] = await Promise.all([api.trainingRun(id), api.metrics(id)]);
  const plots = new MetricPlots("No metrics logged yet.");
  const head = h("div");
  const factsBox = h("div");
  const checkpoints = h("div");
  const step = h("span", { class: "muted" });

  function draw() {
    head.replaceChildren(
      h("p", null, link("/training", icon("back"), "Training")),
      h("div", { class: "toolbar" },
        h("h1", { class: "mono" }, run.id), statusPill(run), step),
    );
    step.textContent = run.step == null ? "" : `step ${run.step} · ${metricsText(run.latest)}`;
    factsBox.replaceChildren(facts(run));
    checkpoints.replaceChildren(checkpointsTable(run.id, run.checkpoints, run.best?.step ?? null));
    plots.draw([{ label: run.id, rows: metrics.rows }]);
  }

  // While it runs, keep it fresh.
  const timer = setInterval(async () => {
    if (statusOf(run) !== "running") return;
    try {
      [run, metrics] = await Promise.all([api.trainingRun(id), api.metrics(id)]);
      draw();
    } catch {
      // Try again next time.
    }
  }, REFRESH_MS);

  draw();
  const files = h("details", { class: "objects" },
    h("summary", null, "Its train.toml and dataset.toml, as they were"),
    h("h3", { class: "mono small" }, run.config.path),
    h("pre", { class: "toml" }, run.config.toml ?? "(none: the model has no train.toml)"),
    h("h3", { class: "mono small" }, run.dataset.path),
    h("pre", { class: "toml" }, run.dataset.toml),
  );
  return {
    element: h("section", { class: "training-run" },
      head, factsBox,
      h("h2", null, "Metrics"), plots.element,
      h("h2", null, "Checkpoints"), checkpoints,
      files,
    ),
    stop: () => {
      clearInterval(timer);
      plots.stop();
    },
  };
}

// ---------------------------------------------------------------------------
// Comparing runs
// ---------------------------------------------------------------------------

export async function compareView(ids: string[]): Promise<TrainingView> {
  if (ids.length < 1) {
    return { element: h("p", null, "Choose runs to compare on the ", link("/training", "training list"), "."), stop: () => {} };
  }
  const runs = await Promise.all(ids.map((id) => api.trainingRun(id)));
  const metrics = await Promise.all(ids.map((id) => api.metrics(id)));
  const plots = new MetricPlots("None of these runs has logged metrics yet.");

  const draw = () => plots.draw(runs.map((r, i) => ({ label: `${r.id} ${r.model}`, rows: metrics[i].rows })));
  const timer = setInterval(async () => {
    if (!runs.some((r) => statusOf(r) === "running")) return;
    try {
      const fresh = await Promise.all(ids.map((id) => api.metrics(id)));
      fresh.forEach((m, i) => (metrics[i] = m));
      draw();
    } catch {
      // Try again next time.
    }
  }, REFRESH_MS);
  draw();

  const legend = h("table", null,
    h("thead", null, h("tr", null, ...["", "Run", "Model", "Who", "Status", "Dataset", "Started from", "Best"].map((t) => h("th", null, t)))),
    h("tbody", null, ...runs.map((r, i) =>
      h("tr", null,
        h("td", null, h("span", { class: "swatch", style: `background:${COLORS[i % COLORS.length]}` })),
        h("td", null, link(`/training/${r.id}`, h("span", { class: "mono" }, r.id))),
        h("td", null, r.model),
        h("td", null, r.trained_by),
        h("td", null, statusPill(r)),
        h("td", { class: "mono small" }, r.dataset.path),
        h("td", { class: "small" }, ...Object.values(r.parents ?? {}).flatMap((p, j) => [j ? ", " : "", runLink(p.checkpoint)])),
        h("td", { class: "mono small" }, r.best ? `step ${r.best.step} ${metricsText(r.best.metrics)}` : ""),
      ))),
  );
  return {
    element: h("section", null,
      h("p", null, link("/training", icon("back"), "Training")),
      h("h1", null, `Comparing ${runs.length} training runs`),
      h("div", { class: "table-wrap" }, legend),
      h("h2", null, "Metrics"),
      plots.element,
    ),
    stop: () => {
      clearInterval(timer);
      plots.stop();
    },
  };
}
