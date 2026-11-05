// Labels drawn over an episode's video from its tables: boxes, full boxes
// (around the whole of a partly hidden object), keypoints (filled if seen,
// hollow if hidden) and names, for the ground truth and for each model
// whose predictions were logged. A canvas over the video, redrawn for the
// frame on screen; a bar of chips turns each layer, label and source on
// and off. Hovering a box or point says what it is.

import { api, Table } from "./api";
import { h } from "./dom";

export type Layer = "boxes" | "full" | "points" | "hidden" | "names";

const LAYERS: [Layer, string, string][] = [
  ["boxes", "Boxes", "Each labeled object's box: what's seen of it"],
  ["full", "Full boxes", "Dashed, around the whole of a partly hidden object"],
  ["points", "Keypoints", "Filled dots: keypoints that are seen"],
  ["hidden", "Hidden keypoints", "Hollow dots: keypoints behind something, or the object's own far side"],
  ["names", "Names", "Each box's label, and a model's confidence"],
];

/** What the viewer chose, kept from one episode to the next. */
export interface OverlayPrefs {
  on: boolean;
  layers: Record<Layer, boolean>;
  hiddenLabels: string[];
  hiddenSources: string[];
}

export const defaultOverlayPrefs = (): OverlayPrefs => ({
  on: true,
  layers: { boxes: true, full: true, points: true, hidden: true, names: true },
  hiddenLabels: [],
  hiddenSources: [],
});

// Ground truth in magenta (as the old drawn-in previews were), models in
// colors that stand out from it and from most scenes.
const TRUTH = "#ff3cc8";
const MODELS = ["#3ad8ff", "#b6f23a", "#ffb340", "#b98cff", "#ff7a59"];
const OUTLINE = "rgba(0, 0, 0, 0.75)";
// In CSS pixels, whatever the video's size.
const POINT_RADIUS = 4;
const HOVER_RADIUS = 7;

interface BoxRow {
  object: number | null;
  label: string;
  x0: number; y0: number; x1: number; y1: number;
  full: [number, number, number, number] | null;
  pixels: number | null;
  fullPixels: number | null;
  score: number | null;
  source: string;
}

interface PointRow {
  object: number | null;
  label: string;
  index: number;
  x: number; y: number;
  visible: boolean | null;
  occludedBy: number | null;
  score: number | null;
  source: string;
}

/** Rows of a column table, grouped by frame. */
function byFrame<T>(table: Table | undefined, keep: (i: number) => boolean, row: (i: number) => T): Map<number, T[]> {
  const out = new Map<number, T[]>();
  if (!table) return out;
  table.frame.forEach((f, i) => {
    if (!keep(i)) return;
    const frame = f as number;
    if (!out.has(frame)) out.set(frame, []);
    out.get(frame)!.push(row(i));
  });
  return out;
}

const num = (v: unknown) => (v === null || v === undefined ? null : Number(v));

export class Overlay {
  /** The video and the canvas over it. */
  stage: HTMLElement;
  /** The chips: put it under the video. */
  bar: HTMLElement;
  private canvas = h("canvas", { class: "overlay" });
  private tip = h("div", { class: "overlay-tip", hidden: true });
  private names = new Map<number, string>();
  private cameraIds = new Map<string, number>();
  private sizes = new Map<number, [number, number]>();
  private boxes = new Map<number, BoxRow[]>();
  private points = new Map<number, PointRow[]>();
  private loaded = new Map<string, Promise<void>>();
  private camera = "";
  private frame = 0;
  private hasFull = false;
  private hasPoints: boolean;
  private labels: string[] = [];
  private sources: string[] = [];
  private failed = "";

  constructor(
    private video: HTMLVideoElement,
    private tables: Record<string, Table>,
    tableNames: string[],
    private keypointsOf: (camera: string) => Promise<Table>,
    private prefs: OverlayPrefs,
  ) {
    this.hasPoints = tableNames.includes("keypoints");
    const objects = tables.objects;
    if (objects) objects.object.forEach((id, i) => this.names.set(id as number, objects.name[i] as string));
    const cameras = tables.cameras;
    if (cameras) {
      cameras.object.forEach((id, i) => {
        this.cameraIds.set(this.names.get(id as number) ?? "", id as number);
        this.sizes.set(id as number, [cameras.width[i] as number, cameras.height[i] as number]);
      });
    }
    const b = tables.boxes;
    if (b) {
      this.hasFull = !!b.full_x0 && b.full_x0.some((v) => v !== null);
      this.labels = [...new Set(b.label as string[])];
      this.sources = [...new Set(b.source as string[])];
    }
    this.stage = h("div", { class: "stage" }, video, this.canvas, this.tip);
    this.bar = h("div", { class: "layers" });
    new ResizeObserver(() => this.redraw()).observe(this.stage);
    video.addEventListener("loadedmetadata", () => this.redraw());
    this.canvas.addEventListener("mousemove", (e) => this.hover(e));
    this.canvas.addEventListener("mouseleave", () => (this.tip.hidden = true));
    this.renderBar();
  }

  /** Which camera's labels to draw. */
  setCamera(camera: string): void {
    this.camera = camera;
    const id = this.cameraIds.get(camera);
    this.boxes = byFrame(this.tables.boxes, (i) => this.tables.boxes.camera[i] === id, (i) => this.boxRow(i));
    this.points = new Map();
    this.loadPoints();
    this.renderBar();
    this.redraw();
  }

  /** Draw the labels of this frame. */
  draw(frame: number): void {
    this.frame = frame;
    this.redraw();
  }

  private boxRow(i: number): BoxRow {
    const t = this.tables.boxes;
    const full = t.full_x0 && t.full_x0[i] !== null
      ? [t.full_x0[i], t.full_y0[i], t.full_x1[i], t.full_y1[i]].map(Number) as [number, number, number, number]
      : null;
    return {
      object: num(t.object[i]), label: t.label[i] as string,
      x0: Number(t.x0[i]), y0: Number(t.y0[i]), x1: Number(t.x1[i]), y1: Number(t.y1[i]),
      full, pixels: num(t.pixels?.[i]), fullPixels: num(t.full_pixels?.[i]), score: num(t.score?.[i]),
      source: t.source[i] as string,
    };
  }

  /** Fetch the camera's keypoints, once, when a keypoint layer is on. */
  private loadPoints(): void {
    const camera = this.camera;
    const { points, hidden } = this.prefs.layers;
    if (!this.hasPoints || !camera || !(points || hidden) || !this.prefs.on) return;
    let loading = this.loaded.get(camera);
    if (!loading) {
      loading = this.keypointsOf(camera).then((t) => {
        const rows = byFrame(t, () => true, (i): PointRow => ({
          object: num(t.object[i]), label: t.label[i] as string, index: Number(t.index[i]),
          x: Number(t.x[i]), y: Number(t.y[i]), visible: t.visible[i] as boolean | null,
          occludedBy: num(t.occluded_by[i]), score: num(t.score?.[i]), source: t.source[i] as string,
        }));
        this.pointsByCamera.set(camera, rows);
        for (const r of [...rows.values()].flat()) {
          if (!this.sources.includes(r.source)) this.sources.push(r.source);
        }
      }).catch((e: Error) => {
        this.failed = `Couldn't load the keypoints: ${e.message}`;
        this.loaded.delete(camera);
      });
      this.loaded.set(camera, loading);
    }
    loading.then(() => {
      if (this.camera !== camera) return;
      this.points = this.pointsByCamera.get(camera) ?? new Map();
      this.renderBar();
      this.redraw();
    });
  }

  private pointsByCamera = new Map<string, Map<number, PointRow[]>>();

  // --- The chips ---------------------------------------------------------

  private renderBar(): void {
    const p = this.prefs;
    const chip = (text: string, on: boolean, title: string, toggle: () => void, swatch?: string) =>
      h("button", {
        class: "chip chip--sm", "aria-pressed": String(on), title,
        onclick: () => { toggle(); this.loadPoints(); this.renderBar(); this.redraw(); },
      }, swatch ? h("span", { class: "swatch", style: `background:${swatch}` }) : null, text);
    const toggleIn = (list: string[], value: string) => {
      const at = list.indexOf(value);
      if (at >= 0) list.splice(at, 1); else list.push(value);
    };

    const has: Record<Layer, boolean> = {
      boxes: !!this.tables.boxes,
      full: this.hasFull,
      points: this.hasPoints,
      hidden: this.hasPoints,
      names: !!this.tables.boxes,
    };
    const master = h("label", { class: "check", title: "Show labels over the video (L)" },
      h("input", {
        type: "checkbox", checked: p.on,
        onchange: (e: Event) => { p.on = (e.target as HTMLInputElement).checked; this.loadPoints(); this.renderBar(); this.redraw(); },
      }),
      " Labels");
    const groups: HTMLElement[] = [];
    if (p.on) {
      groups.push(h("div", { class: "layer-group" },
        ...LAYERS.filter(([layer]) => has[layer]).map(([layer, text, title]) =>
          chip(text, p.layers[layer], title, () => (p.layers[layer] = !p.layers[layer])))));
      if (this.labels.length > 1) {
        groups.push(h("div", { class: "layer-group" },
          h("span", { class: "layer-title" }, "Labels"),
          ...this.labels.map((label) => chip(label, !p.hiddenLabels.includes(label), `Show ${label}`,
            () => toggleIn(p.hiddenLabels, label)))));
      }
      if (this.sources.length > 1) {
        groups.push(h("div", { class: "layer-group" },
          h("span", { class: "layer-title" }, "From"),
          ...this.sources.map((source) => chip(source === "truth" ? "Ground truth" : source,
            !p.hiddenSources.includes(source), source === "truth" ? "The simulation's exact labels" : `${source}'s predictions`,
            () => toggleIn(p.hiddenSources, source), this.color(source)))));
      }
    }
    const note = this.failed ? h("span", { class: "error small" }, this.failed) : null;
    const anything = Object.values(has).some(Boolean);
    this.bar.hidden = !anything;
    this.bar.replaceChildren(master, ...groups, note ?? "");
  }

  /** Turn every label on or off (the L key). */
  toggle(): void {
    this.prefs.on = !this.prefs.on;
    this.loadPoints();
    this.renderBar();
    this.redraw();
  }

  private color(source: string): string {
    if (source === "truth") return TRUTH;
    const models = this.sources.filter((s) => s !== "truth");
    return MODELS[Math.max(0, models.indexOf(source)) % MODELS.length];
  }

  // --- Drawing -----------------------------------------------------------

  /** Picture pixels to CSS pixels: the video is drawn at its element's size. */
  private scale(): [number, number] {
    const id = this.cameraIds.get(this.camera);
    const [w, h_] = this.video.videoWidth
      ? [this.video.videoWidth, this.video.videoHeight]
      : this.sizes.get(id ?? -1) ?? [1, 1];
    return [this.canvas.clientWidth / w, this.canvas.clientHeight / h_];
  }

  private shownBoxes(): BoxRow[] {
    const p = this.prefs;
    return (this.boxes.get(this.frame) ?? []).filter((b) =>
      !p.hiddenLabels.includes(b.label) && !p.hiddenSources.includes(b.source));
  }

  private shownPoints(): PointRow[] {
    const p = this.prefs;
    const rows = this.points.get(this.frame) ?? [];
    return rows.filter((r) => !p.hiddenSources.includes(r.source)
      && (r.object === null || !this.objectHidden(r.object))
      && (r.visible === false ? p.layers.hidden : p.layers.points));
  }

  /** Points belong to objects, not labels; an object's label is its boxes'. */
  private objectHidden(object: number): boolean {
    const objects = this.tables.objects;
    if (!objects || !this.prefs.hiddenLabels.length) return false;
    const i = objects.object.indexOf(object);
    const label = i >= 0 ? objects.label[i] as string | null : null;
    return !!label && this.prefs.hiddenLabels.includes(label);
  }

  private redraw(): void {
    const dpr = window.devicePixelRatio || 1;
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(cw * dpr) || this.canvas.height !== Math.round(ch * dpr)) {
      this.canvas.width = Math.round(cw * dpr);
      this.canvas.height = Math.round(ch * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const p = this.prefs;
    if (!p.on || !this.camera) return;
    const [sx, sy] = this.scale();
    const boxes = this.shownBoxes();

    ctx.lineJoin = "round";
    if (p.layers.full) {
      ctx.setLineDash([5, 4]);
      for (const b of boxes) {
        if (!b.full || !differs(b)) continue;
        strokeRect(ctx, b.full[0] * sx, b.full[1] * sy, (b.full[2] - b.full[0]) * sx, (b.full[3] - b.full[1]) * sy,
          this.color(b.source), 1.5);
      }
      ctx.setLineDash([]);
    }
    if (p.layers.boxes) {
      for (const b of boxes) {
        strokeRect(ctx, b.x0 * sx, b.y0 * sy, (b.x1 - b.x0) * sx, (b.y1 - b.y0) * sy, this.color(b.source), 2);
      }
    }
    if (p.layers.points || p.layers.hidden) {
      for (const r of this.shownPoints()) {
        const x = r.x * sx, y = r.y * sy;
        ctx.beginPath();
        ctx.arc(x, y, POINT_RADIUS, 0, 2 * Math.PI);
        if (r.visible === false) {
          ctx.lineWidth = 3.5;
          ctx.strokeStyle = OUTLINE;
          ctx.stroke();
          ctx.lineWidth = 1.75;
          ctx.strokeStyle = this.color(r.source);
          ctx.stroke();
        } else {
          ctx.fillStyle = this.color(r.source);
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = OUTLINE;
          ctx.stroke();
        }
      }
    }
    if (p.layers.names && (p.layers.boxes || p.layers.full)) {
      ctx.font = "600 11px system-ui, sans-serif";
      ctx.textBaseline = "bottom";
      for (const b of boxes) {
        const text = b.source === "truth" || b.score === null ? b.label : `${b.label} ${b.score.toFixed(2)}`;
        const w = ctx.measureText(text).width + 8;
        const x = Math.min(Math.max(0, b.x0 * sx), cw - w);
        const y = Math.max(16, b.y0 * sy);
        ctx.fillStyle = this.color(b.source);
        ctx.fillRect(x, y - 16, w, 16);
        ctx.fillStyle = "#000";
        ctx.fillText(text, x + 4, y - 2);
      }
    }
  }

  // --- Hovering ----------------------------------------------------------

  private hover(e: MouseEvent): void {
    if (!this.prefs.on) return;
    const [sx, sy] = this.scale();
    const mx = e.offsetX, my = e.offsetY;
    const name = (id: number | null) => (id === null ? "" : this.names.get(id) ?? `object ${id}`);
    let text = "";
    const p = this.prefs;
    if (p.layers.points || p.layers.hidden) {
      let best: PointRow | null = null;
      let bestD = HOVER_RADIUS;
      for (const r of this.shownPoints()) {
        const d = Math.hypot(r.x * sx - mx, r.y * sy - my);
        if (d <= bestD) { best = r; bestD = d; }
      }
      if (best) {
        const seen = best.visible === false
          ? best.occludedBy === null ? "hidden"
            : best.occludedBy === best.object ? "hidden by its own far side" : `hidden by ${name(best.occludedBy)}`
          : best.visible === null ? "" : "seen";
        const who = best.object === null ? best.source : name(best.object);
        text = [`${who} · ${best.label} ${best.index}`, seen, best.score !== null ? `score ${best.score.toFixed(2)}` : ""]
          .filter(Boolean).join(" · ");
      }
    }
    if (!text && (p.layers.boxes || p.layers.full)) {
      const inside = this.shownBoxes().filter((b) => {
        const [x0, y0, x1, y1] = p.layers.full && b.full ? b.full : [b.x0, b.y0, b.x1, b.y1];
        return mx >= x0 * sx && mx <= x1 * sx && my >= y0 * sy && my <= y1 * sy;
      });
      // The smallest box under the pointer: usually the one meant.
      inside.sort((a, b) => (a.x1 - a.x0) * (a.y1 - a.y0) - (b.x1 - b.x0) * (b.y1 - b.y0));
      const b = inside[0];
      if (b) {
        const share = b.pixels && b.fullPixels ? Math.min(1, b.pixels / b.fullPixels) : null;
        text = b.source === "truth"
          ? [`${name(b.object)} · ${b.label}`, share !== null ? `${Math.round(share * 100)}% seen` : "",
              b.pixels !== null ? `${b.pixels} px` : ""].filter(Boolean).join(" · ")
          : `${b.source} · ${b.label}${b.score !== null ? ` ${b.score.toFixed(2)}` : ""}`;
      }
    }
    this.tip.hidden = !text;
    if (!text) return;
    this.tip.textContent = text;
    // Beside the pointer, kept inside the video.
    const w = this.tip.offsetWidth, hgt = this.tip.offsetHeight;
    const left = mx + 14 + w > this.canvas.clientWidth ? mx - 14 - w : mx + 14;
    const top = Math.min(Math.max(0, my - hgt - 8), this.canvas.clientHeight - hgt);
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${top}px`;
  }
}

/** A full box worth drawing: more than a pixel from the box seen. */
function differs(b: BoxRow): boolean {
  if (!b.full) return false;
  const [x0, y0, x1, y1] = b.full;
  return Math.max(Math.abs(x0 - b.x0), Math.abs(y0 - b.y0), Math.abs(x1 - b.x1), Math.abs(y1 - b.y1)) > 1;
}

function strokeRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, hgt: number, color: string,
                    width: number): void {
  // A dark edge under the color, so it reads on light and dark scenes.
  ctx.lineWidth = width + 2;
  ctx.strokeStyle = OUTLINE;
  ctx.strokeRect(x, y, w, hgt);
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, w, hgt);
}

export const keypointsFrom = (runId: string, index: number) => (camera: string) =>
  api.keypoints(runId, index, camera);
