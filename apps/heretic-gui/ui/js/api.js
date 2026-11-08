// SPDX-License-Identifier: AGPL-3.0-or-later
// Backend commands and formatting helpers.

import { invoke } from "./tauri.js";

export const api = {
  appInfo: () => invoke("app_info"),
  presetIds: () => invoke("preset_ids"),
  presetSettings: (id) => invoke("preset_settings", { id }),
  validateSettings: (settings) => invoke("validate_settings", { settings }),
  settingsToToml: (settings) => invoke("settings_to_toml", { settings }),
  settingsFromToml: (text) => invoke("settings_from_toml", { text }),
  searchModels: (query) => invoke("search_models", { query }),
  inspectModel: (id) => invoke("inspect_model", { id }),
  listProjects: (checkpointDir) => invoke("list_projects", { checkpointDir }),
  deleteProject: (path) => invoke("delete_project", { path }),
  checkpointInfo: (settings) => invoke("checkpoint_info", { settings }),
  startRun: (settings, checkpointAction) => invoke("start_run", { settings, checkpointAction }),
  runMoreTrials: (count) => invoke("run_more_trials", { count }),
  stopRun: () => invoke("stop_run"),
  cancelRun: () => invoke("cancel_run"),
  getTrials: () => invoke("get_trials"),
  restoreTrial: (number) => invoke("restore_trial", { number }),
  chatSend: (messages, useAdapters) => invoke("chat_send", { messages, useAdapters }),
  chatStop: () => invoke("chat_stop"),
  saveModel: (directory, strategy) => invoke("save_model", { directory, strategy }),
  saveGguf: (directory, ggufType, llamaCppDir) => invoke("save_gguf", { directory, ggufType, llamaCppDir }),
  checkLlamaCpp: (directory) => invoke("check_llama_cpp", { directory }),
  hfWhoami: (token) => invoke("hf_whoami", { token }),
  uploadModel: (token, repoId, isPrivate, strategy) =>
    invoke("upload_model", { token, repoId, private: isPrivate, strategy }),
  modelCardPreview: () => invoke("model_card_preview"),
  forgetToken: () => invoke("forget_token"),
  currentVersion: () => invoke("update_current_version"),
  checkUpdate: (includePrereleases) => invoke("check_update", { includePrereleases }),
  installUpdate: (request) => invoke("install_update", { request }),
  openReleasePage: (url) => invoke("open_release_page", { url }),
};

/** Formats a byte count for humans. */
export function formatBytes(bytes) {
  if (bytes == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}

/** Formats a duration like the CLI ("1h 5m", "3m 20s", "42s"). */
export function formatDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

/** Removes `**bold**` markers. */
export function plain(text) {
  return String(text ?? "").replaceAll("**", "");
}

/** Rough memory estimate (bytes) for running a model with the given dtype. */
export function estimateVram(weightsBytes, checkpointDtype, dtype) {
  if (weightsBytes == null) return null;
  const size = (name) => (name === "float32" ? 4 : name === "float64" ? 8 : 2);
  const resolved = dtype === "auto" ? checkpointDtype : dtype;
  const weights = (weightsBytes / size(checkpointDtype)) * size(resolved);
  // Activations, KV cache and LoRA adapters need extra room.
  return weights * 1.25 + 0.75 * 1024 ** 3;
}

/** Pareto-optimal trials sorted by objective values, like the CLI. */
export function paretoFront(trials) {
  return trials
    .filter((t) => t.pareto_optimal)
    .sort((a, b) => {
      for (let i = 0; i < Math.min(a.values.length, b.values.length); i++) {
        if (a.values[i] !== b.values[i]) return a.values[i] - b.values[i];
      }
      return 0;
    });
}
