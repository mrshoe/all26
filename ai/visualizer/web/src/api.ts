// The server's API (../../server/app.py).

import { idToken } from "./auth";

export class SignInNeeded extends Error {}

export interface Me {
  email: string;
  handle: string;
}

/** Labeled photos imported as a run (ai/src/shared/photos.py), not simulated. */
export interface PhotosInfo {
  name: string;
  url?: string;
  license?: string;
  credit?: string;
  labels: string[];
  splits: string[];
}

export interface SimInfo {
  scenario: string;
  scene: string;
  backend: string;
}

export interface Run {
  id: string;
  launched_by: string;
  created_at: string;
  /** Not for imported photos, which ran nothing. */
  command?: string[];
  cwd?: string;
  episodes: number;
  gpu?: boolean;
  git?: { commit: string; branch: string; dirty: boolean };
  sim?: SimInfo;
  photos?: PhotosInfo;
  /** Set when the job is a training run's (shared/training.py marks it). */
  training?: string;
  status_counts: Record<string, number>;
}

export interface EpisodeRecord {
  index: number;
  seed?: number;
  status: string;
  exit_code?: number;
  started_at?: string;
  finished_at?: string;
  sim?: SimInfo & { frames: number; fps: number; cameras: string[]; labels: string[] };
  photos?: { split: string; frames: number; fps: number; cameras: string[]; labels: string[] };
}

export interface RunDetail extends Run {
  episodes_detail: EpisodeRecord[];
}

/** A table as columns: {column: values}. */
export type Table = Record<string, (number | string | boolean | null)[]>;

export interface EpisodeData {
  run: string;
  index: number;
  /** 1: from before the tables (only videos); 2: the tables below. */
  format: number;
  info: {
    // Simulations.
    scenario?: string;
    scene?: string;
    seed?: number;
    // Imported photos: where they're from, and which of its splits.
    photos_from?: string;
    split?: string;
    fps: number;
    frames: number;
    tables?: string[];
    // Format 1 only.
    objects?: Record<string, Record<string, unknown>>;
  };
  /** objects, cameras, sim, poses, boxes … (format 2; empty for format 1). */
  tables: Record<string, Table>;
  /** Each camera's video; `boxes`, with boxes drawn in, only for format 1
   *  (later episodes' labels are drawn over the plain one, from the tables). */
  previews: Record<string, { plain: string; boxes?: string }>;
}

// Training runs (ai/src/shared/training.py writes them).

export interface Git {
  commit: string;
  branch: string;
  dirty: boolean;
}

export interface Where {
  kind: "laptop" | "cloud";
  host?: string;
  run?: string;
  episode?: number;
}

export interface TrainingSummary {
  id: string;
  model: string;
  trained_by: string;
  created_at: string;
  finished_at?: string | null;
  heartbeat_at?: string;
  status: string;
  error?: string | null;
  where: Where;
  step: number | null;
  latest: Record<string, number>;
  best: { step: number; metrics: Record<string, number> } | null;
  /** Results that aren't on the training curve (run.final), e.g. the test split's. */
  final?: Record<string, number>;
  last_checkpoint: number | null;
  checkpoints: number;
  dataset: { path: string; samples: Record<string, number> };
  /** Each [start] name's checkpoint, "<run>/<step>". */
  parents: Record<string, string>;
  /** How many times it was resumed. */
  resumed: number;
}

export interface Checkpoint {
  step: number;
  file: string;
  size: number;
  saved_at: string;
  resumable: boolean;
  best: boolean;
  pruned?: boolean;
  tags: string[];
  metrics: Record<string, number>;
}

export interface TrainingDetail extends Omit<TrainingSummary, "parents" | "dataset" | "resumed" | "checkpoints"> {
  segments: { source: string; source_sha256?: string; git?: Git; started_at: string; where?: Where; resumed_at_step?: number }[];
  config: { path: string; toml: string | null };
  dataset: { path: string; toml: string; samples: Record<string, number>; episodes: Record<string, number> };
  parents: Record<string, { ref: string; checkpoint: string; model: string }>;
  checkpoints: Checkpoint[];
  /** What run.keep() kept, each with a signed link. */
  files: KeptFile[];
  children: { id: string; model: string; trained_by: string; status: string; heartbeat_at?: string }[];
}

export interface KeptFile {
  name: string;
  size: number;
  content_type: string;
  kept_at?: string;
  url: string;
}

/** One log() call: its step, time, and metrics. */
export type MetricRow = { step: number; time: number } & Record<string, number>;

async function get<T>(path: string): Promise<T> {
  const token = await idToken();
  if (!token) throw new SignInNeeded();
  const response = await fetch(path, { headers: { Authorization: `Bearer ${token}` } });
  if (response.status === 401) throw new SignInNeeded();
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail ?? `The server said HTTP ${response.status}.`);
  return data as T;
}

export const api = {
  config: () => fetch("/api/config").then((r) => r.json() as Promise<{ project_id: string; firebase_api_key: string }>),
  me: () => get<Me>("/api/me"),
  runs: (opts: { user?: string; sim?: boolean }) => {
    const q = new URLSearchParams({ limit: "100" });
    if (opts.user) q.set("user", opts.user);
    if (opts.sim) q.set("sim", "true");
    return get<Run[]>(`/api/runs?${q}`);
  },
  run: (id: string) => get<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  episode: (id: string, n: number) => get<EpisodeData>(`/api/runs/${encodeURIComponent(id)}/episodes/${n}`),
  /** One camera's keypoints that are in the picture. */
  keypoints: (id: string, n: number, camera: string) =>
    get<Table>(`/api/runs/${encodeURIComponent(id)}/episodes/${n}/keypoints?camera=${encodeURIComponent(camera)}`),
  trainingRuns: (opts: { model?: string; user?: string }) => {
    const q = new URLSearchParams({ limit: "100" });
    if (opts.model) q.set("model", opts.model);
    if (opts.user) q.set("user", opts.user);
    return get<TrainingSummary[]>(`/api/training?${q}`);
  },
  trainingRun: (id: string) => get<TrainingDetail>(`/api/training/${encodeURIComponent(id)}`),
  metrics: (id: string) => get<{ run: string; rows: MetricRow[] }>(`/api/training/${encodeURIComponent(id)}/metrics`),
};
