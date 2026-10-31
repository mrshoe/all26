// An episode's tables (shared/logger.py) as series to plot: one value per
// frame, for a field of a table, for one object (or for the whole episode,
// in tables like `sim` that have no objects).
//
// Each series is shown as <table>.<camera>.<object>.<field>, with only the
// parts its table has:
//
//   poses.ball.z               the ball's height
//   boxes.front.ball.height    the ball's box height in the front camera
//                              (ground truth; width, height, cx and cy are
//                              worked out from x0 y0 x1 y1)
//   sim.render_s               seconds spent rendering each frame
//
// Object names can hold dots (walls.0), so the key the plots keep is
// table|subject|field, with camera/object as a boxes subject.

import { Table } from "./api";

export type Column = (number | null)[];

const KEYS = new Set(["frame", "object", "camera"]);
const BOX_EXTRA = ["width", "height", "cx", "cy"];

export interface Subject {
  key: string; // "" for tables without objects
  label: string;
}

export function seriesKey(table: string, subject: string, field: string): string {
  return `${table}|${subject}|${field}`;
}

export class Catalog {
  readonly time: number[];
  readonly names = new Map<number, string>();
  private static = new Set<number>();
  private cache = new Map<string, Column>();

  constructor(private tables: Record<string, Table>, readonly frames: number, fps: number) {
    const objects = tables.objects;
    if (objects) {
      objects.object.forEach((id, i) => {
        this.names.set(id as number, objects.name[i] as string);
        if (objects.static[i]) this.static.add(id as number);
      });
    }
    // Each frame's time, from the first table with frame and time columns.
    const timed = Object.values(tables).find((t) => t.frame && t.time && !t.object && !t.camera);
    this.time = Array.from({ length: frames }, (_, f) => f / fps);
    if (timed) timed.frame.forEach((f, i) => (this.time[f as number] = timed.time[i] as number));
  }

  /** Tables with something to plot: a frame column, and not the pictures. */
  plottable(): string[] {
    return Object.keys(this.tables).filter((t) => this.tables[t].frame && t !== "images");
  }

  subjects(table: string): Subject[] {
    const t = this.tables[table];
    if (!t) return [];
    if (table === "boxes") {
      const seen = new Map<string, Subject>();
      t.object.forEach((obj, i) => {
        if (obj === null || t.source[i] !== "truth") return;
        const camera = this.name(t.camera[i] as number);
        const name = this.name(obj as number);
        const key = `${camera}/${name}`;
        if (!seen.has(key)) seen.set(key, { key, label: `${camera}.${name}` });
      });
      return [...seen.values()].sort((a, b) => a.label.localeCompare(b.label));
    }
    if (t.object) {
      // Moving things first: those are the ones worth plotting.
      const ids = [...new Set(t.object as number[])];
      ids.sort((a, b) => Number(this.static.has(a)) - Number(this.static.has(b)) || this.name(a).localeCompare(this.name(b)));
      return ids.map((id) => ({ key: this.name(id), label: this.name(id) + (this.static.has(id) ? " (static)" : "") }));
    }
    return [{ key: "", label: "" }];
  }

  fields(table: string): string[] {
    const t = this.tables[table];
    if (!t) return [];
    const numeric = Object.keys(t).filter((c) => !KEYS.has(c) && t[c].some((v) => typeof v === "number"));
    return table === "boxes" ? [...numeric, ...BOX_EXTRA] : numeric;
  }

  label(key: string): string {
    const [table, subject, field] = key.split("|");
    return [table, ...subject.split("/").filter(Boolean), field].join(".");
  }

  has(key: string): boolean {
    const [table, subject, field] = key.split("|");
    return this.fields(table).includes(field) && this.subjects(table).some((s) => s.key === subject);
  }

  values(key: string): Column {
    let out = this.cache.get(key);
    if (!out) {
      out = this.compute(key);
      this.cache.set(key, out);
    }
    return out;
  }

  private name(id: number): string {
    return this.names.get(id) ?? String(id);
  }

  private compute(key: string): Column {
    const [table, subject, field] = key.split("|");
    const t = this.tables[table];
    const out: Column = new Array(this.frames).fill(null);
    if (!t) return out;
    let rows = t.frame.map((_, i) => i);
    if (table === "boxes") {
      const [camera, name] = subject.split("/");
      rows = rows.filter((i) => t.source[i] === "truth" && this.name(t.camera[i] as number) === camera
        && t.object[i] !== null && this.name(t.object[i] as number) === name);
    } else if (t.object) {
      rows = rows.filter((i) => this.name(t.object[i] as number) === subject);
    }
    const get = (c: string, i: number) => t[c][i] as number | null;
    const value = (i: number): number | null => {
      if (table === "boxes" && BOX_EXTRA.includes(field)) {
        const [x0, y0, x1, y1] = ["x0", "y0", "x1", "y1"].map((c) => get(c, i)!);
        return { width: x1 - x0, height: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 }[field]!;
      }
      return get(field, i);
    };
    for (const i of rows) {
      const f = t.frame[i] as number;
      if (f >= 0 && f < this.frames) out[f] = value(i);
    }
    // Static objects are logged once, at frame 0, and stay there.
    if (table === "poses" && rows.length === 1 && out[0] !== null) out.fill(out[0]);
    return out;
  }
}
