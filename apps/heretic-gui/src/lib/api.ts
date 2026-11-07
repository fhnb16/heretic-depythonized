// SPDX-License-Identifier: AGPL-3.0-or-later
// Typed wrappers around the backend commands.

import { invoke } from "@tauri-apps/api/core";

export type Settings = Record<string, any>;

export interface Score {
  value: number;
  display: string;
  md_display: string;
}

export interface ScoreRecord {
  name: string;
  score: Score;
  baseline: Score;
}

export interface TrialSummary {
  number: number;
  index: number;
  values: number[];
  scores: ScoreRecord[];
  parameters: [string, string][];
  pareto_optimal: boolean;
}

export interface TrialsResponse {
  trials: TrialSummary[];
  objective_names: string[];
  baseline: [string, Score][];
  applied_trial: number | null;
  model: string;
  modified_model_type: string;
  modifier: string | null;
}

export interface AppInfo {
  version: string;
  cuda_compiled: boolean;
  hf_cache: string;
  has_hf_token: boolean;
  default_checkpoint_dir: string;
}

export interface ModelSearchResult {
  id: string;
  downloads: number;
  likes: number;
}

export interface ModelInspection {
  id: string;
  supported: boolean;
  message: string | null;
  model_type: string | null;
  architectures: string[];
  layers: number | null;
  multimodal: boolean;
  checkpoint_dtype: string | null;
  weights_bytes: number | null;
  has_chat_template: boolean;
}

export interface Project {
  model: string;
  path: string;
  finished: boolean;
  completed_trials: number;
  n_trials: number | null;
  modified: number | null;
  settings: Settings | null;
}

export interface CheckpointInfo {
  path: string;
  finished: boolean;
  completed_trials: number;
  n_trials: number | null;
}

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface HubUser {
  name: string;
  fullname: string | null;
  email: string | null;
}

export const api = {
  appInfo: () => invoke<AppInfo>("app_info"),
  presetIds: () => invoke<string[]>("preset_ids"),
  presetSettings: (id: string) => invoke<Settings>("preset_settings", { id }),
  settingsSchema: () => invoke<any>("get_settings_schema"),
  validateSettings: (settings: Settings) => invoke<string[]>("validate_settings", { settings }),
  settingsToToml: (settings: Settings) => invoke<string>("settings_to_toml", { settings }),
  settingsFromToml: (text: string) => invoke<Settings>("settings_from_toml", { text }),
  searchModels: (query: string) => invoke<ModelSearchResult[]>("search_models", { query }),
  inspectModel: (id: string) => invoke<ModelInspection>("inspect_model", { id }),
  listProjects: (checkpointDir: string) => invoke<Project[]>("list_projects", { checkpointDir }),
  deleteProject: (path: string) => invoke<void>("delete_project", { path }),
  checkpointInfo: (settings: Settings) => invoke<CheckpointInfo | null>("checkpoint_info", { settings }),
  startRun: (settings: Settings, checkpointAction: "continue" | "restart" | null) =>
    invoke<void>("start_run", { settings, checkpointAction }),
  runMoreTrials: (count: number) => invoke<void>("run_more_trials", { count }),
  stopRun: () => invoke<void>("stop_run"),
  cancelRun: () => invoke<void>("cancel_run"),
  getTrials: () => invoke<TrialsResponse>("get_trials"),
  restoreTrial: (number: number) => invoke<{ applied_trial: number }>("restore_trial", { number }),
  chatSend: (messages: Message[], useAdapters: boolean) =>
    invoke<{ response: string }>("chat_send", { messages, useAdapters }),
  chatStop: () => invoke<void>("chat_stop"),
  saveModel: (directory: string, strategy: "merge" | "adapter") =>
    invoke<{ directory: string; files: string[] }>("save_model", { directory, strategy }),
  hfWhoami: (token: string | null) => invoke<HubUser>("hf_whoami", { token }),
  uploadModel: (token: string | null, repoId: string, isPrivate: boolean, strategy: "merge" | "adapter") =>
    invoke<{ repo_id: string }>("upload_model", { token, repoId, private: isPrivate, strategy }),
  modelCardPreview: () => invoke<string>("model_card_preview"),
  forgetToken: () => invoke<void>("forget_token"),
};

/** Formats a byte count for humans. */
export function formatBytes(bytes: number | null | undefined): string {
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

/** Formats a duration in seconds like the CLI does ("1h 5m", "3m 20s", "42s"). */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

/** Removes `**bold**` markers from backend text. */
export function plain(text: string): string {
  return text.replaceAll("**", "");
}

/** Splits backend text into plain and bold segments. */
export function segments(text: string): { text: string; bold: boolean }[] {
  return text.split("**").map((part, index) => ({ text: part, bold: index % 2 === 1 }));
}

/** Rough VRAM estimate (bytes) for running a model with the given dtype. */
export function estimateVram(weightsBytes: number | null, checkpointDtype: string | null, dtype: string): number | null {
  if (weightsBytes == null) return null;
  const size = (name: string | null) => (name === "float32" ? 4 : name === "float64" ? 8 : 2);
  const resolved = dtype === "auto" ? checkpointDtype : dtype;
  const weights = (weightsBytes / size(checkpointDtype)) * size(resolved);
  // Activations, KV cache and LoRA adapters need extra room.
  return weights * 1.25 + 0.75 * 1024 ** 3;
}
