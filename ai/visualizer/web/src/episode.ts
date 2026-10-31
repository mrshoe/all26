// One episode: a camera's video, with or without boxes, and plots of its
// signals with a playhead that follows the video. Click a plot to jump there.
// It fills the run page's main pane (runs.ts).

import uPlot from "uplot";
import { api, EpisodeData } from "./api";
import { h } from "./dom";

const SPEEDS = [0.1, 0.25, 0.5, 1, 2];

/** What the viewer chose, kept from one episode to the next. */
export interface Prefs {
  camera?: string;
  boxes: boolean;
  speed: number;
  plots?: string[][];
}

export const defaultPrefs = (): Prefs => ({ boxes: true, speed: 1 });

type Signals = Record<string, (number | null)[]>;

/** The logged signals, plus box sizes and centers worked out from the boxes. */
function withDerived(signals: Signals): Signals {
  const out: Signals = { ...signals };
  for (const key of Object.keys(signals)) {
    if (!key.endsWith(".x0")) continue;
    const base = key.slice(0, -3);
    const [x0, y0, x1, y1] = ["x0", "y0", "x1", "y1"].map((c) => signals[`${base}.${c}`]);
    if (!x0 || !y0 || !x1 || !y1) continue;
    const each = (f: (i: number) => number) =>
      x0.map((_, i) => (x0[i] === null || y0[i] === null || x1[i] === null || y1[i] === null ? null : f(i)));
    out[`${base}.width`] = each((i) => x1[i]! - x0[i]!);
    out[`${base}.height`] = each((i) => y1[i]! - y0[i]!);
    out[`${base}.cx`] = each((i) => (x0[i]! + x1[i]!) / 2);
    out[`${base}.cy`] = each((i) => (y0[i]! + y1[i]!) / 2);
  }
  return out;
}

const COLORS = ["#2a7de1", "#e1572a", "#2aa876", "#a14ad9", "#d9a20f", "#d94a8c"];

class Plot {
  element: HTMLElement;
  private plot: uPlot | null = null;
  private chosen: string[];
  private holder = h("div", { class: "plot" });

  constructor(
    private signals: Signals,
    private time: number[],
    chosen: string[],
    private playhead: () => number,
    private seek: (t: number) => void,
    remove: () => void,
  ) {
    this.chosen = chosen.filter((s) => s in signals);
    const names = Object.keys(signals).filter((s) => s !== "frame" && s !== "time");
    const picker = h("details", { class: "picker" },
      h("summary", null, "Signals"),
      h("div", { class: "choices" },
        ...names.map((name) =>
          h("label", null,
            h("input", {
              type: "checkbox",
              checked: this.chosen.includes(name),
              onchange: (e: Event) => {
                const on = (e.target as HTMLInputElement).checked;
                this.chosen = on ? [...this.chosen, name] : this.chosen.filter((s) => s !== name);
                this.draw();
              },
            }),
            " ", name),
        ),
      ),
    );
    this.element = h("div", { class: "plot-card" },
      h("div", { class: "plot-bar" }, picker, h("button", { class: "linkish", onclick: remove }, "Remove plot")),
      this.holder,
    );
    new ResizeObserver(() => this.plot?.setSize({ width: this.width(), height: 220 })).observe(this.holder);
  }

  private width(): number {
    return Math.max(300, this.holder.clientWidth);
  }

  /** The signals it shows. */
  shown(): string[] {
    return [...this.chosen];
  }

  draw(): void {
    this.plot?.destroy();
    this.plot = null;
    if (!this.chosen.length) {
      this.holder.replaceChildren(h("p", { class: "muted" }, "Pick signals to plot."));
      return;
    }
    this.holder.replaceChildren();
    const opts: uPlot.Options = {
      width: this.width(),
      height: 220,
      scales: { x: { time: false } },
      axes: [{ label: "seconds", stroke: "currentColor" }, { stroke: "currentColor" }],
      series: [
        { label: "t", value: (_u, v) => (v == null ? "" : v.toFixed(2)) },
        ...this.chosen.map((name, i) => ({
          label: name,
          stroke: COLORS[i % COLORS.length],
          width: 2,
          spanGaps: false,
          value: (_u: uPlot, v: number | null) => (v == null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)),
        })),
      ],
      cursor: { drag: { x: false, y: false } },
      hooks: {
        draw: [
          (u) => {
            const x = u.valToPos(this.playhead(), "x", true);
            const ctx = u.ctx;
            ctx.save();
            ctx.strokeStyle = "#e11d48";
            ctx.lineWidth = 2 * devicePixelRatio;
            ctx.beginPath();
            ctx.moveTo(x, u.bbox.top);
            ctx.lineTo(x, u.bbox.top + u.bbox.height);
            ctx.stroke();
            ctx.restore();
          },
        ],
      },
    };
    const data = [this.time, ...this.chosen.map((name) => this.signals[name])] as uPlot.AlignedData;
    this.plot = new uPlot(opts, data, this.holder);
    this.plot.over.addEventListener("click", (e) => {
      const t = this.plot!.posToVal((e as MouseEvent).offsetX, "x");
      this.seek(Math.max(0, t));
    });
  }

  refresh(): void {
    this.plot?.redraw(false, false);
  }

  destroy(): void {
    this.plot?.destroy();
  }
}

export interface EpisodeView {
  element: HTMLElement;
  /** Stop playing, and save the viewer's choices to the Prefs. */
  stop: () => void;
}

export async function episodeView(runId: string, index: number, prefs: Prefs): Promise<EpisodeView> {
  const ep: EpisodeData = await api.episode(runId, index);
  const info = ep.info;
  const fps = info.fps;
  const cameras = Object.keys(ep.previews);
  const signals = withDerived(ep.signals);
  const time = (signals.time ?? []) as number[];
  const labeled = Object.keys(info.labels);

  let camera = prefs.camera && cameras.includes(prefs.camera) ? prefs.camera : cameras[0];
  let boxes = prefs.boxes;
  const video = h("video", { muted: true, playsInline: true, preload: "auto", class: "video" });
  const source = () => (camera ? (boxes ? ep.previews[camera].boxes : ep.previews[camera].plain) : "");
  video.src = source();
  video.defaultPlaybackRate = video.playbackRate = prefs.speed;

  const frameOf = () => Math.min(info.frames - 1, Math.floor(video.currentTime * fps + 1e-6));
  const playhead = () => frameOf() / fps;
  const seek = (t: number) => {
    // The middle of a frame, so the video shows that frame and not its neighbor.
    const frame = Math.min(info.frames - 1, Math.max(0, Math.round(t * fps)));
    video.currentTime = (frame + 0.5) / fps;
    update();
  };

  const plots: Plot[] = [];
  const plotArea = h("div", { class: "plots" });
  const addPlot = (chosen: string[]) => {
    const plot: Plot = new Plot(signals, time, chosen, playhead, seek, () => {
      plot.destroy();
      plot.element.remove();
      plots.splice(plots.indexOf(plot), 1);
    });
    plots.push(plot);
    plotArea.append(plot.element);
    plot.draw();
  };

  const scrub = h("input", {
    type: "range", min: 0, max: info.frames - 1, step: 1, value: 0, class: "scrub",
    oninput: (e: Event) => seek(Number((e.target as HTMLInputElement).value) / fps),
  });
  const readout = h("span", { class: "mono readout" });
  const playButton = h("button", { onclick: () => (video.paused ? video.play() : video.pause()) }, "Play");
  const speed = h("select", {
    onchange: (e: Event) => {
      prefs.speed = video.playbackRate = Number((e.target as HTMLSelectElement).value);
    },
  }, ...SPEEDS.map((s) => h("option", { value: s, selected: s === prefs.speed }, `${s}×`)));

  function update() {
    const frame = frameOf();
    scrub.value = String(frame);
    readout.textContent = `frame ${frame + 1}/${info.frames} · ${(frame / fps).toFixed(2)} s`;
    playButton.textContent = video.paused ? "Play" : "Pause";
    for (const plot of plots) plot.refresh();
  }

  let frameRequest = 0;
  const tick = () => {
    update();
    if (!video.paused) frameRequest = requestAnimationFrame(tick);
  };
  video.addEventListener("play", () => (frameRequest = requestAnimationFrame(tick)));
  video.addEventListener("pause", update);
  video.addEventListener("seeked", update);
  video.addEventListener("loadedmetadata", update);

  const swap = () => {
    const at = video.currentTime;
    const playing = !video.paused;
    video.src = source();
    video.addEventListener("loadedmetadata", () => {
      video.playbackRate = prefs.speed;
      video.currentTime = at;
      if (playing) video.play();
    }, { once: true });
  };

  const cameraPicker = cameras.length > 1
    ? h("select", { onchange: (e: Event) => { prefs.camera = camera = (e.target as HTMLSelectElement).value; swap(); } },
        ...cameras.map((c) => h("option", { value: c, selected: c === camera }, c)))
    : h("span", null, camera ? `Camera: ${camera}` : "No cameras");
  const boxToggle = h("label", null,
    h("input", { type: "checkbox", checked: boxes, onchange: (e: Event) => { prefs.boxes = boxes = (e.target as HTMLInputElement).checked; swap(); } }),
    " Boxes");

  const keys = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    if (e.key === "ArrowRight") { video.pause(); seek((frameOf() + 1) / fps); }
    if (e.key === "ArrowLeft") { video.pause(); seek((frameOf() - 1) / fps); }
  };
  document.addEventListener("keydown", keys);

  const objects = Object.entries(info.objects);
  const element = h("section", { class: "episode" },
    h("h1", null, `Episode ${index}`),
    h("p", { class: "muted" }, `${info.scenario} in ${info.scene} · seed `, h("span", { class: "mono" }, String(info.seed)),
      ` · ${info.frames} frames at ${fps} fps`),
    h("div", { class: "player" },
      h("div", { class: "viewer" },
        h("div", { class: "toolbar" }, cameraPicker, boxToggle),
        video,
        h("div", { class: "controls" },
          h("button", { onclick: () => { video.pause(); seek((frameOf() - 1) / fps); }, title: "Previous frame (←)" }, "◀"),
          playButton,
          h("button", { onclick: () => { video.pause(); seek((frameOf() + 1) / fps); }, title: "Next frame (→)" }, "▶"),
          scrub, speed, readout,
        ),
      ),
      h("div", { class: "side" },
        plotArea,
        h("button", { onclick: () => addPlot([]) }, "+ Add plot"),
      ),
    ),
    h("details", { class: "objects" },
      h("summary", null, `What was drawn for this episode (${objects.length} objects)`),
      h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Object"), h("th", null, "Type"), h("th", null, "Settings"))),
        h("tbody", null, ...objects.map(([name, o]) => {
          const { type, id: _id, label, ...rest } = o as Record<string, unknown>;
          return h("tr", null,
            h("td", null, name, label ? h("span", { class: "tag" }, String(label)) : null),
            h("td", null, String(type)),
            h("td", { class: "mono small" }, JSON.stringify(rest)));
        })),
      ),
    ),
  );

  if (prefs.plots) {
    prefs.plots.forEach(addPlot);
  } else {
    // Two plots to start: where the labeled objects are, and how big their
    // boxes are in the camera.
    addPlot(labeled.map((o) => `${o}.z`));
    if (camera) addPlot(labeled.map((o) => `${camera}.${o}.height`));
  }
  update();

  return {
    element,
    stop: () => {
      prefs.plots = plots.map((p) => p.shown());
      cancelAnimationFrame(frameRequest);
      document.removeEventListener("keydown", keys);
      video.pause();
      plots.forEach((p) => p.destroy());
    },
  };
}
