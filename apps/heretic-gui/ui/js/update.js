// SPDX-License-Identifier: AGPL-3.0-or-later
// Update check against the GitHub releases (backend: src-tauri/src/update.rs).

import { api, formatBytes } from "./api.js";
import { h, progressBar } from "./dom.js";
import { prefs, set, setPref, state, t } from "./store.js";
import { listen } from "./tauri.js";

/** Label of the update button: installable builds update in place, others open the release. */
export function installLabel() {
  return state.update?.install === "browser" ? t("updateDownload") : t("updateInstall");
}

/** Download and install progress, or null when nothing is being installed. */
export function installProgress() {
  if (state.updateBusy !== "installing") return null;
  const p = state.updateProgress;
  const stage = p?.stage ?? "downloading";
  const label = {
    downloading: t("updateDownloading"),
    verifying: t("updateVerifying"),
    installing: t("updateInstalling"),
  }[stage];
  const downloading = stage === "downloading" && p?.total;
  return h(
    "div",
    { class: "stack tight" },
    h(
      "span",
      { class: "row" },
      h("span", { class: "badge accent" }, label),
      downloading
        ? h(
            "span",
            { class: "num faint small", title: `${formatBytes(p.done)} / ${formatBytes(p.total)}` },
            `${Math.floor((p.done / p.total) * 100)}%`,
          )
        : null,
    ),
    progressBar(downloading ? p.done / p.total : null),
  );
}

/** Silent checks run at most this often. */
const AUTO_INTERVAL_MS = 6 * 60 * 60 * 1000;

/** Whether the shell should show the update notice. */
export function showNotice() {
  const update = state.update;
  return !!update?.available && update.tag !== prefs.updateDismissed && state.updateBusy !== "checking";
}

/** Checks for an update; `silent` checks keep errors out of the interface. */
export async function checkForUpdates({ silent = false } = {}) {
  if (state.updateBusy) return;
  set({ updateBusy: "checking", updateError: null });
  try {
    const update = await api.checkUpdate(prefs.updatePrereleases);
    setPref("updateLastCheck", Date.now());
    setPref("updateCache", update);
    set({ update, updateBusy: null });
  } catch (error) {
    set({ updateBusy: null, updateError: silent ? null : String(error) });
    if (silent) console.warn("Update check failed:", error);
  }
}

/** Postpones the notice until a newer release appears. */
export function dismissUpdate() {
  setPref("updateDismissed", state.update?.tag ?? null);
}

/** Downloads and starts the update, or opens the release page. */
export async function installUpdate() {
  const update = state.update;
  if (!update?.available || state.updateBusy) return;
  set({ updateBusy: "installing", updateError: null, updateProgress: null });
  try {
    const outcome = await api.installUpdate({
      asset: update.asset,
      checksums_url: update.checksums_url,
      install: update.install,
      url: update.url,
    });
    // "restarting": the backend exits the app in a moment; keep the busy state until then.
    if (outcome !== "restarting") set({ updateBusy: null, updateProgress: null });
  } catch (error) {
    set({ updateBusy: null, updateProgress: null, updateError: String(error) });
  }
}

export async function openReleaseNotes() {
  if (!state.update?.url) return;
  try {
    await api.openReleasePage(state.update.url);
  } catch (error) {
    set({ updateError: String(error) });
  }
}

/** Startup: restores the last result for this build and checks again if it is due. */
export async function initUpdates() {
  await listen("heretic://update", (event) => set({ updateProgress: event.payload }));
  let current = null;
  try {
    current = await api.currentVersion();
  } catch (error) {
    console.warn(error);
  }
  // A cached result only holds for the build that produced it (after an update it is stale).
  const fresh = prefs.updateCache && prefs.updateCache.current === current;
  if (fresh) set({ update: prefs.updateCache });
  if (prefs.updateAuto && (!fresh || Date.now() - (prefs.updateLastCheck || 0) > AUTO_INTERVAL_MS)) {
    await checkForUpdates({ silent: true });
  }
}
