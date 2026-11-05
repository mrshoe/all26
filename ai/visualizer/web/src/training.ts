// Training runs (ai/src/shared/training.py records them):
//
//   /training                     the list; tick runs to compare them
//   /training/<id>                one run: its facts, a table of its metrics,
//                                 a plot per metric (live while it runs),
//                                 its checkpoints and kept files
//   /training/compare?runs=a,b    those runs' metrics, overlaid
//
// Metrics are plotted against step. A metric logged only now and then (a
// validation score, say) is drawn through the steps it has.
//
// Metric names say their split by prefix: train_box_loss, val_box_loss and
// test_box_loss are one metric, box_loss, on three splits. The table and
// plots group them that way, losses first. Other metrics (lr...) follow,
// and ones that are always equal (lr_pg0, lr_pg1, lr_pg2) show as one.

import uPlot from "uplot";
import { api, Checkpoint, KeptFile, Me, MetricRow, TrainingDetail, Where } from "./api";
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

/** Where it ran: a laptop, or a cloud job (linking to the job's run page). */
function whereNode(where?: Where): (Node | string)[] {
  if (!where) return [];
  if (where.kind !== "cloud") return [`laptop ${where.host ?? ""}`];
  return ["cloud run ", link(`/runs/${where.run}`, h("span", { class: "mono" }, where.run ?? "")), `, episode ${where.episode ?? 0}`];
}

function metricsText(metrics?: Record<string, number> | null): string {
  return Object.entries(metrics ?? {}).map(([k, v]) => `${k} ${format(v)}`).join("  ");
}

/** For the list: what the best checkpoint was saved with, and final results. */
function resultText(run: { best: { step: number; metrics: Record<string, number> } | null; final?: Record<string, number> }): string {
  const parts = [];
  if (run.best && Object.keys(run.best.metrics ?? {}).length) parts.push(`best ${metricsText(run.best.metrics)} @ ${run.best.step}`);
  if (run.final && Object.keys(run.final).length) parts.push(`final ${metricsText(run.final)}`);
  return parts.join(" · ");
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
// Organizing metrics: by split, losses first
// ---------------------------------------------------------------------------

const SPLITS = ["train", "val", "test"] as const;
type Split = (typeof SPLITS)[number];
const PREFIXES: [string, Split][] = [["train_", "train"], ["validation_", "val"], ["val_", "val"], ["test_", "test"]];
const SPLIT_COLORS: Record<Split, string> = { train: COLORS[0], val: COLORS[1], test: COLORS[2] };

/** A metric across splits: box_loss is train_box_loss, val_box_loss... */
interface SplitMetric {
  base: string;
  names: Partial<Record<Split, string>>;
}

/** A metric with no split. `names` are those folded into it: always equal. */
interface OtherMetric {
  base: string;
  names: string[];
}

interface Organized {
  split: SplitMetric[];
  other: OtherMetric[];
}

function splitOf(name: string): [Split, string] | null {
  for (const [prefix, split] of PREFIXES) if (name.startsWith(prefix) && name.length > prefix.length) return [split, name.slice(prefix.length)];
  return null;
}

/** Every metric the rows (and final results) hold, in the order first seen. */
function metricNames(rows: MetricRow[][], final: Record<string, number>[] = []): string[] {
  const names = new Set<string>();
  for (const r of rows) for (const row of r) for (const k of Object.keys(row)) if (k !== "step" && k !== "time") names.add(k);
  for (const f of final) for (const k of Object.keys(f)) names.add(k);
  return [...names];
}

/** Group metrics by split, losses first, and fold always-equal ones. */
function organize(rows: MetricRow[][], final: Record<string, number>[] = []): Organized {
  const bySplit = new Map<string, SplitMetric>();
  const others: string[] = [];
  for (const name of metricNames(rows, final)) {
    const found = splitOf(name);
    if (!found) {
      others.push(name);
      continue;
    }
    const [split, base] = found;
    if (!bySplit.has(base)) bySplit.set(base, { base, names: {} });
    bySplit.get(base)!.names[split] = name;
  }
  const isLoss = (base: string) => /loss/.test(base);
  const split = [...bySplit.values()];
  split.sort((a, b) => Number(isLoss(b.base)) - Number(isLoss(a.base)));

  // lr_pg0, lr_pg1, lr_pg2: one "lr" when they're equal wherever logged.
  const groups = new Map<string, string[]>();
  for (const name of others) {
    const base = name.replace(/_?(pg)?\d+$/, "") || name;
    groups.set(base, [...(groups.get(base) ?? []), name]);
  }
  const other: OtherMetric[] = [];
  for (const [base, names] of groups) {
    const equal = names.length > 1 && rows.every((r) => r.every((row) => names.every((n) => row[n] === row[names[0]])));
    if (equal) other.push({ base, names });
    else other.push(...names.map((n) => ({ base: n, names: [n] })));
  }
  return { split, other };
}

/** A metric's last value, and its best: lowest for losses, highest otherwise. */
function latestAndBest(rows: MetricRow[], name: string): { latest?: number; best?: number; bestStep?: number } {
  const lower = /loss/.test(name);
  let latest: number | undefined;
  let best: number | undefined;
  let bestStep: number | undefined;
  for (const row of rows) {
    const v = row[name];
    if (v === undefined || !Number.isFinite(v)) continue;
    latest = v;
    if (best === undefined || (lower ? v < best : v > best)) {
      best = v;
      bestStep = row.step;
    }
  }
  return { latest, best, bestStep };
}

function metricsTable(rows: MetricRow[], final: Record<string, number>): HTMLElement {
  const { split, other } = organize([rows], [final]);
  if (!split.length && !other.length) return h("p", { class: "muted" }, "No metrics logged yet.");
  const splits = SPLITS.filter((sp) => split.some((m) => m.names[sp]));
  // Best only means something for a split's metric (not lr, say).
  const cell = (name: string | undefined, showBest = true) => {
    if (!name) return h("td", { class: "num muted" }, "–");
    if (name in final) return h("td", { class: "num" }, format(final[name]), h("span", { class: "tag", title: "run.final(): not on the training curve" }, "final"));
    const { latest, best, bestStep } = latestAndBest(rows, name);
    if (latest === undefined) return h("td", { class: "num muted" }, "–");
    return h("td", { class: "num" },
      h("span", { class: "metric-value" }, format(latest)),
      showBest && best !== undefined && format(best) !== format(latest) ? h("span", { class: "metric-best" }, `best ${format(best)} @ ${bestStep}`) : null);
  };
  return h("div", { class: "table-wrap" },
    h("table", { class: "metrics-table" },
      h("thead", null, h("tr", null, h("th", null, "Metric"), ...splits.map((sp) => h("th", { class: "num" }, sp)))),
      h("tbody", null,
        ...split.map((m) => h("tr", null, h("td", { class: "mono" }, m.base), ...splits.map((sp) => cell(m.names[sp])))),
        ...other.map((m) => h("tr", { class: "metric-other" },
          h("td", { class: "mono", title: m.names.length > 1 ? `${m.names.join(", ")}: always equal` : undefined }, m.base),
          // No split: its value spans the columns.
          h("td", { class: "num", colspan: Math.max(1, splits.length) }, ...[cell(m.names[0], false)].flatMap((c) => [...c.childNodes])))),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// Plots: one panel per metric, each line a split or a run
// ---------------------------------------------------------------------------

interface Line {
  label: string;
  rows: MetricRow[];
  key: string;
  color: string;
}

interface Panel {
  title: string;
  lines: Line[];
}

/** One run: a panel per metric, its splits' lines together (train against
 * val shows overfitting at a glance). */
function runPanels(rows: MetricRow[]): Panel[] {
  const { split, other } = organize([rows]);
  return [
    ...split.map((m) => ({
      title: m.base,
      lines: SPLITS.filter((sp) => m.names[sp] && rows.some((r) => m.names[sp]! in r))
        .map((sp) => ({ label: sp, rows, key: m.names[sp]!, color: SPLIT_COLORS[sp] })),
    })),
    ...other.map((m) => ({ title: m.base, lines: [{ label: m.base, rows, key: m.names[0], color: COLORS[3] }] })),
  ].filter((p) => p.lines.length);
}

/** Several runs: a panel per metric name, a line per run. */
function comparePanels(series: { label: string; rows: MetricRow[] }[]): Panel[] {
  const { split, other } = organize(series.map((s) => s.rows));
  const names = [...split.flatMap((m) => SPLITS.map((sp) => m.names[sp]).filter((n): n is string => !!n)),
                 ...other.map((m) => m.names[0])];
  return names.map((name) => ({
    title: other.find((m) => m.names[0] === name && m.names.length > 1)?.base ?? name,
    lines: series.map((s, i) => ({ label: s.label, rows: s.rows, key: name, color: COLORS[i % COLORS.length] }))
      .filter((l) => l.rows.some((r) => name in r)),
  })).filter((p) => p.lines.length);
}

class MetricPlots {
  element = h("div", { class: "metric-grid" });
  private plots: uPlot[] = [];
  private observer = new ResizeObserver(() => this.resize());

  constructor(private emptyText: string) {
    this.observer.observe(this.element);
  }

  draw(panels: Panel[]): void {
    this.destroy();
    if (!panels.length) {
      this.element.replaceChildren(h("p", { class: "muted" }, this.emptyText));
      return;
    }
    this.element.replaceChildren();
    for (const panel of panels) {
      // Every line's steps, so they line up; each line's values where it has them.
      const steps = [...new Set(panel.lines.flatMap((l) => l.rows.filter((r) => l.key in r).map((r) => r.step)))].sort((a, b) => a - b);
      const data: (number | null)[][] = [steps];
      for (const l of panel.lines) {
        const at = new Map<number, number>();
        for (const r of l.rows) if (l.key in r) at.set(r.step, r[l.key]);
        data.push(steps.map((step) => at.get(step) ?? null));
      }
      const holder = h("div", { class: "plot" });
      const card = h("div", { class: "plot-card" }, h("h3", { class: "metric-name mono" }, panel.title), holder);
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
          ...panel.lines.map((l) => ({
            label: l.label,
            stroke: l.color,
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
        h("td", { class: "mono small" }, resultText(run)),
        h("td", { class: "small" }, ...whereNode(run.where)),
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
          ...["", "Run", "Model", "Who", "When", "Status", "Step", "Result", "Where"].map((t) => h("th", null, t)))),
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
    rows.push([i === 0 ? "Code" : `Resumed at step ${s.resumed_at_step}`, [`${git} · `, ...whereNode(s.where ?? run.where)]]);
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

/** What run.keep() kept: pictures shown, other files linked. */
function keptFiles(files: KeptFile[]): HTMLElement[] {
  if (!files.length) return [];
  const pictures = files.filter((f) => f.content_type.startsWith("image/"));
  const others = files.filter((f) => !f.content_type.startsWith("image/"));
  const parts: (HTMLElement | null)[] = [
    h("h2", null, "Files"),
    pictures.length ? h("div", { class: "kept-grid" }, ...pictures.map((f) =>
      h("figure", { class: "kept" },
        h("a", { href: f.url, target: "_blank", rel: "noopener" }, h("img", { src: f.url, alt: f.name, loading: "lazy" })),
        h("figcaption", { class: "mono small" }, f.name)))) : null,
    others.length ? h("ul", { class: "kept-list" }, ...others.map((f) =>
      h("li", null, h("a", { href: f.url, target: "_blank", rel: "noopener", class: "mono" }, f.name),
        h("span", { class: "muted small" }, ` ${bytes(f.size)}`)))) : null,
  ];
  return parts.filter((e): e is HTMLElement => e !== null);
}

export async function trainingRunView(id: string): Promise<TrainingView> {
  let [run, metrics] = await Promise.all([api.trainingRun(id), api.metrics(id)]);
  const plots = new MetricPlots("No metrics logged yet.");
  const head = h("div");
  const factsBox = h("div");
  const table = h("div");
  const checkpoints = h("div");
  const kept = h("div");
  const step = h("span", { class: "muted" });

  function draw() {
    head.replaceChildren(
      h("p", null, link("/training", icon("back"), "Training")),
      h("div", { class: "toolbar" },
        h("h1", { class: "mono" }, run.id), statusPill(run), step),
    );
    step.textContent = run.step == null ? "" : `step ${run.step}`;
    factsBox.replaceChildren(facts(run));
    table.replaceChildren(metricsTable(metrics.rows, run.final ?? {}));
    checkpoints.replaceChildren(checkpointsTable(run.id, run.checkpoints, run.best?.step ?? null));
    kept.replaceChildren(...keptFiles(run.files ?? []));
    plots.draw(runPanels(metrics.rows));
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
      h("h2", null, "Metrics"), table, plots.element,
      kept,
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

  const draw = () => plots.draw(comparePanels(runs.map((r, i) => ({ label: `${r.id} ${r.model}`, rows: metrics[i].rows }))));
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
