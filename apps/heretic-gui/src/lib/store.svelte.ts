// SPDX-License-Identifier: AGPL-3.0-or-later
// Global application state and backend event handling.

import { listen } from "@tauri-apps/api/event";
import { api, type Settings, type TrialSummary, type Score } from "./api";
import { detectLanguage, translate, type Key, type Lang } from "./i18n";

export type View = "home" | "new" | "run" | "results" | "chat" | "settings";
export type RunStatus = "idle" | "preparing" | "running" | "ready" | "failed";
export type Phase =
  | "loading_model"
  | "batch_size"
  | "response_prefix"
  | "scorers"
  | "modifier"
  | "optimizing"
  | "finished"
  | "restoring_trial"
  | "exporting";
export type Theme = "system" | "light" | "dark";

export interface LogLine {
  level: "heading" | "info" | "detail" | "success" | "warning" | "error" | "debug";
  text: string;
  time: number;
}

function load<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(`heretic.${key}`);
    return value == null ? fallback : (JSON.parse(value) as T);
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown) {
  try {
    localStorage.setItem(`heretic.${key}`, JSON.stringify(value));
  } catch {
    // Storage may be unavailable; preferences then only last for this session.
  }
}

export const prefs = $state({
  lang: load<Lang>("lang", detectLanguage()),
  theme: load<Theme>("theme", "system"),
  // Filled from the backend's default on first start.
  checkpointDir: load<string>("checkpointDir", ""),
});

export function setPref<K extends keyof typeof prefs>(key: K, value: (typeof prefs)[K]) {
  prefs[key] = value;
  save(key, value);
  if (key === "theme") applyTheme();
}

export function applyTheme() {
  const root = document.documentElement;
  if (prefs.theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", prefs.theme);
}

export function t(key: Key, params: Record<string, string | number> = {}): string {
  return translate(prefs.lang, key, params);
}

export const app = $state({
  view: "home" as View,
  status: "idle" as RunStatus,
  statusMessage: null as string | null,
  outcome: null as string | null,
  phase: null as Phase | null,
  logs: [] as LogLine[],
  progress: null as { task: string; done: number; total: number | null } | null,
  trials: [] as TrialSummary[],
  objectiveNames: [] as string[],
  baseline: [] as [string, Score][],
  total: 0,
  current: null as { index: number; parameters: [string, string][] } | null,
  elapsed: 0,
  remaining: null as number | null,
  settings: null as Settings | null,
  appliedTrial: null as number | null,
  modelLoaded: false,
  chatStream: "",
  error: null as string | null,
});

const MAX_LOG_LINES = 2000;

export function pushLog(level: LogLine["level"], text: string) {
  app.logs.push({ level, text, time: Date.now() });
  if (app.logs.length > MAX_LOG_LINES) app.logs.splice(0, app.logs.length - MAX_LOG_LINES);
}

/** Resets the per-run state before a new run starts. */
export function resetRun(settings: Settings) {
  app.status = "preparing";
  app.statusMessage = null;
  app.outcome = null;
  app.phase = null;
  app.logs = [];
  app.progress = null;
  app.trials = [];
  app.objectiveNames = [];
  app.baseline = [];
  app.total = settings.n_trials ?? 0;
  app.current = null;
  app.elapsed = 0;
  app.remaining = null;
  app.settings = settings;
  app.appliedTrial = null;
  app.modelLoaded = false;
}

export async function refreshTrials() {
  try {
    const response = await api.getTrials();
    app.trials = response.trials;
    app.objectiveNames = response.objective_names;
    app.baseline = response.baseline;
    app.appliedTrial = response.applied_trial;
  } catch (error) {
    app.error = String(error);
  }
}

let initialized = false;

/** Subscribes to backend events (once). */
export async function initEvents() {
  if (initialized) return;
  initialized = true;

  await listen<{ level: LogLine["level"]; text: string }>("heretic://log", (event) => {
    pushLog(event.payload.level, event.payload.text);
  });

  await listen<{ task: string; done: number; total: number | null }>("heretic://progress", (event) => {
    const p = event.payload;
    app.progress = p.total != null && p.done >= p.total ? null : p;
  });

  await listen<{ state: RunStatus; message: string | null; outcome: string | null }>("heretic://status", (event) => {
    const { state, message, outcome } = event.payload;
    app.status = state;
    app.statusMessage = message;
    app.outcome = outcome;
    if (state === "running" || state === "ready") app.modelLoaded = true;
    if (state === "ready") {
      app.progress = null;
      app.current = null;
      refreshTrials();
    }
    if (state === "failed") app.progress = null;
  });

  await listen<any>("heretic://session", (event) => {
    const e = event.payload;
    switch (e.type) {
      case "phase":
        app.phase = e.phase;
        break;
      case "baseline":
        app.baseline = e.scores;
        app.objectiveNames = e.objective_names;
        break;
      case "trial_started":
        app.current = { index: e.index, parameters: e.parameters };
        app.total = e.total;
        break;
      case "trial_finished": {
        const trial: TrialSummary = e.trial;
        const pareto = new Set<number>(e.pareto);
        app.trials = [...app.trials.filter((t) => t.number !== trial.number), trial].map((t) => ({
          ...t,
          pareto_optimal: pareto.has(t.number),
        }));
        app.total = e.total;
        app.elapsed = e.elapsed_seconds;
        app.remaining = e.remaining_seconds;
        app.appliedTrial = trial.number;
        break;
      }
      case "trial_cancelled":
        app.current = null;
        break;
    }
  });

  await listen<{ text: string }>("heretic://chat", (event) => {
    app.chatStream += event.payload.text;
  });
}

/** Pareto-optimal trials sorted by objective values, like the CLI. */
export function paretoFront(trials: TrialSummary[]): TrialSummary[] {
  return trials
    .filter((t) => t.pareto_optimal)
    .sort((a, b) => {
      for (let i = 0; i < Math.min(a.values.length, b.values.length); i++) {
        if (a.values[i] !== b.values[i]) return a.values[i] - b.values[i];
      }
      return 0;
    });
}
