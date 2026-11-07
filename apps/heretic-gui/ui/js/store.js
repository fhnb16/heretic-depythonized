// SPDX-License-Identifier: AGPL-3.0-or-later
// Global state, preferences and backend event handling.

import { api } from "./api.js";
import { detectLanguage, translate } from "./i18n.js";
import { listen } from "./tauri.js";
import { DEFAULT_THEME, THEME_VARIABLES, themeVariables } from "./themes.js";

function load(key, fallback) {
  try {
    const value = localStorage.getItem(`heretic.${key}`);
    return value == null ? fallback : JSON.parse(value);
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(`heretic.${key}`, JSON.stringify(value));
  } catch {
    // Storage may be unavailable; preferences then only last for this session.
  }
}

export const prefs = {
  lang: load("lang", detectLanguage()),
  // Mode: system | light | dark.
  theme: load("theme", "system"),
  // Color theme: "heretic" (the app's own) or a catalog id from themes.js.
  skin: load("skin", DEFAULT_THEME.id),
  // Filled from the backend's default on first start.
  checkpointDir: load("checkpointDir", ""),
};

export function t(key, params) {
  return translate(prefs.lang, key, params);
}

const darkQuery = window.matchMedia?.("(prefers-color-scheme: dark)");

export function isDark() {
  return prefs.theme === "dark" || (prefs.theme === "system" && !!darkQuery?.matches);
}

export function applyTheme() {
  const root = document.documentElement;
  const dark = isDark();
  root.setAttribute("data-theme", dark ? "dark" : "light");
  for (const name of THEME_VARIABLES) root.style.removeProperty(name);
  for (const [name, value] of Object.entries(themeVariables(prefs.skin, dark))) root.style.setProperty(name, value);
}

darkQuery?.addEventListener?.("change", () => {
  if (prefs.theme === "system") {
    applyTheme();
    notify();
  }
});

export const state = {
  view: "home",
  status: "idle", // idle | preparing | running | ready | failed
  statusMessage: null,
  phase: null,
  logs: [],
  logGeneration: 0, // incremented when the log is cleared
  progress: null,
  trials: [],
  objectiveNames: [],
  baseline: [],
  total: 0,
  current: null,
  elapsed: 0,
  remaining: null,
  settings: null,
  appliedTrial: null,
  modelLoaded: false,
  chatStream: "",
  info: null,
};

const subscribers = new Set();
let scheduled = false;

/** Calls `fn(state)` after every batch of changes; returns an unsubscribe function. */
export function subscribe(fn) {
  subscribers.add(fn);
  return () => subscribers.delete(fn);
}

export function notify() {
  if (scheduled) return;
  scheduled = true;
  // A timer instead of requestAnimationFrame, which is paused while the window is minimized.
  setTimeout(() => {
    scheduled = false;
    for (const fn of [...subscribers]) {
      try {
        fn(state);
      } catch (error) {
        console.error(error);
      }
    }
  }, 16);
}

export function set(patch) {
  Object.assign(state, patch);
  notify();
}

export function setPref(key, value) {
  prefs[key] = value;
  save(key, value);
  if (key === "theme" || key === "skin") applyTheme();
  notify();
}

export const isBusy = () => state.status === "preparing" || state.status === "running";

const MAX_LOG_LINES = 3000;

function pushLog(level, text) {
  state.logs.push({ level, text });
  if (state.logs.length > MAX_LOG_LINES) {
    state.logs.splice(0, state.logs.length - MAX_LOG_LINES);
    state.logGeneration++;
  }
  notify();
}

/** Resets the per-run state before a run starts. */
export function resetRun(settings) {
  set({
    status: "preparing",
    statusMessage: null,
    phase: null,
    logs: [],
    logGeneration: state.logGeneration + 1,
    progress: null,
    trials: [],
    objectiveNames: [],
    baseline: [],
    total: settings.n_trials ?? 0,
    current: null,
    elapsed: 0,
    remaining: null,
    settings,
    appliedTrial: null,
    modelLoaded: false,
  });
}

export async function refreshTrials() {
  try {
    const response = await api.getTrials();
    set({
      trials: response.trials,
      objectiveNames: response.objective_names,
      baseline: response.baseline,
      appliedTrial: response.applied_trial,
    });
  } catch (error) {
    console.error(error);
  }
}

/** Subscribes to backend events. */
export async function initEvents() {
  await listen("heretic://log", (event) => pushLog(event.payload.level, event.payload.text));

  await listen("heretic://progress", (event) => {
    const p = event.payload;
    set({ progress: p.total != null && p.done >= p.total ? null : p });
  });

  await listen("heretic://status", (event) => {
    const { state: status, message } = event.payload;
    const patch = { status, statusMessage: message };
    if (status === "running" || status === "ready") patch.modelLoaded = true;
    if (status === "ready" || status === "failed") {
      patch.progress = null;
      patch.current = null;
    }
    set(patch);
    if (status === "ready") refreshTrials();
  });

  await listen("heretic://session", (event) => {
    const e = event.payload;
    switch (e.type) {
      case "phase":
        set({ phase: e.phase });
        break;
      case "baseline":
        set({ baseline: e.scores, objectiveNames: e.objective_names });
        break;
      case "trial_started":
        set({ current: { index: e.index, parameters: e.parameters }, total: e.total });
        break;
      case "trial_finished": {
        const pareto = new Set(e.pareto);
        const trials = [...state.trials.filter((t) => t.number !== e.trial.number), e.trial].map((t) => ({
          ...t,
          pareto_optimal: pareto.has(t.number),
        }));
        set({
          trials,
          total: e.total,
          elapsed: e.elapsed_seconds,
          remaining: e.remaining_seconds,
          appliedTrial: e.trial.number,
        });
        break;
      }
      case "trial_cancelled":
        set({ current: null });
        break;
    }
  });

  await listen("heretic://chat", (event) => {
    state.chatStream += event.payload.text;
    notify();
  });
}

/** Applies GUI-wide settings (checkpoint folder) to run settings. */
export function withAppSettings(settings) {
  return { ...settings, study_checkpoint_dir: prefs.checkpointDir || settings.study_checkpoint_dir };
}

/** Starts a run and switches to the run view. */
export async function launch(settings, checkpointAction) {
  const prepared = withAppSettings(settings);
  resetRun(prepared);
  set({ view: "run" });
  try {
    await api.startRun(prepared, checkpointAction);
  } catch (error) {
    set({ status: "failed", statusMessage: String(error) });
  }
}
