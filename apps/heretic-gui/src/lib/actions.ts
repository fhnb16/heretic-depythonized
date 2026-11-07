// SPDX-License-Identifier: AGPL-3.0-or-later
// Actions shared by several views.

import { api, type Settings } from "./api";
import { app, prefs, resetRun } from "./store.svelte";

/** Applies GUI-wide settings (checkpoint folder) to run settings. */
export function withAppSettings(settings: Settings): Settings {
  return { ...settings, study_checkpoint_dir: prefs.checkpointDir || settings.study_checkpoint_dir };
}

/** Starts a run and switches to the run view. */
export async function launch(settings: Settings, checkpointAction: "continue" | "restart" | null) {
  const prepared = withAppSettings(settings);
  resetRun(prepared);
  app.view = "run";
  try {
    await api.startRun(prepared, checkpointAction);
  } catch (error) {
    app.status = "failed";
    app.statusMessage = String(error);
  }
}
