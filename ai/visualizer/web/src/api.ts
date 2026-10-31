// The server's API (../../server/app.py).

import { idToken } from "./auth";

export class SignInNeeded extends Error {}

export interface Me {
  email: string;
  handle: string;
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
  command: string[];
  cwd: string;
  episodes: number;
  gpu: boolean;
  git?: { commit: string; branch: string; dirty: boolean };
  sim?: SimInfo;
  status_counts: Record<string, number>;
}

export interface EpisodeRecord {
  index: number;
  seed: number;
  status: string;
  exit_code?: number;
  started_at?: string;
  finished_at?: string;
  sim?: SimInfo & { frames: number; fps: number; cameras: string[]; labels: string[] };
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
    scenario: string;
    scene: string;
    seed: number;
    fps: number;
    frames: number;
    tables?: string[];
    // Format 1 only.
    objects?: Record<string, Record<string, unknown>>;
  };
  /** objects, cameras, sim, poses, boxes … (format 2; empty for format 1). */
  tables: Record<string, Table>;
  previews: Record<string, { plain: string; boxes: string }>;
}

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
};
