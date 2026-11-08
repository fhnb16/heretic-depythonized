// SPDX-License-Identifier: AGPL-3.0-or-later

import { api } from "../api.js";
import { h, replace } from "../dom.js";
import { icon } from "../icons.js";
import { isDark, prefs, setPref, state, subscribe, t } from "../store.js";
import { openDialog } from "../tauri.js";
import { DEFAULT_THEME, THEMES, themePreview } from "../themes.js";
import { checkForUpdates, installLabel, installProgress, installUpdate, openReleaseNotes } from "../update.js";
import { LLAMA_CPP_RELEASES, getLlamaCppDir, setLlamaCppDir } from "../llamacpp.js";

/** A tile with a miniature of the theme in the current mode. */
function themeTile(theme, active, onpick) {
  const c = themePreview(theme.id, isDark());
  const card = `background: ${c.surface}; border-color: ${c.line}; border-radius: ${c.radius}`;
  return h(
    "button",
    { class: `theme-tile${active ? " active" : ""}`, title: theme.name, "aria-pressed": String(active), onclick: onpick },
    h(
      "span",
      { class: "swatch", style: `background: ${c.bg}` },
      h(
        "span",
        { class: "swatch-side", style: `background: ${c.side}; border-color: ${c.line}` },
        h("span", { class: "swatch-nav", style: `background: ${c.accent}` }),
        h("span", { class: "swatch-nav", style: `background: ${c.muted}` }),
        h("span", { class: "swatch-nav", style: `background: ${c.muted}` }),
      ),
      h(
        "span",
        { class: "swatch-main" },
        h(
          "span",
          { class: "swatch-card", style: card },
          h("span", { class: "swatch-line", style: `background: ${c.text}` }),
          h("span", { class: "swatch-line short", style: `background: ${c.muted}` }),
          h("span", { class: "swatch-button", style: `background: ${c.accent}; border-radius: ${c.buttonRadius}` }),
        ),
      ),
    ),
    h("span", { class: "name", style: c.font ? `font-family: ${c.font}` : "" }, theme.name),
  );
}

export function mount(root) {
  const container = h("div", { class: "settings stack-lg" });
  root.append(container);
  const folder = h("input", { type: "text" });
  folder.addEventListener("change", () => setPref("checkpointDir", folder.value.trim()));
  let forgotten = false;

  const segmented = (current, options, onpick) =>
    h(
      "div",
      { class: "segmented" },
      options.map(([value, label]) => h("button", { class: current === value ? "active" : "", onclick: () => onpick(value) }, label)),
    );

  function render() {
    folder.value = prefs.checkpointDir;
    const info = state.info;
    replace(
      container,
      h("h1", null, t("navSettings")),
      h(
        "section",
        { class: "card stack" },
        h("h3", null, t("settingsAppearance")),
        h(
          "div",
          { class: "setting-row" },
          h("span", null, t("settingsLanguage")),
          segmented(
            prefs.lang,
            [
              ["ru", "Русский"],
              ["en", "English"],
            ],
            (lang) => {
              setPref("lang", lang);
              // Labels are rendered once per view, so rebuild the whole interface.
              window.dispatchEvent(new Event("heretic:relabel"));
            },
          ),
        ),
        h(
          "div",
          { class: "setting-row" },
          h("span", null, t("settingsMode")),
          segmented(
            prefs.theme,
            [
              ["system", t("themeSystem")],
              ["light", t("themeLight")],
              ["dark", t("themeDark")],
            ],
            (theme) => {
              setPref("theme", theme);
              render();
            },
          ),
        ),
        h("span", null, t("settingsTheme")),
        h(
          "div",
          { class: "themes" },
          [DEFAULT_THEME, ...THEMES].map((theme) =>
            themeTile(theme, prefs.skin === theme.id, () => {
              setPref("skin", theme.id);
              render();
            }),
          ),
        ),
        h("p", { class: "faint small" }, t("settingsThemeHint")),
      ),
      h(
        "section",
        { class: "card stack" },
        h("h3", null, t("settingsStorage")),
        h(
          "label",
          { class: "field" },
          t("settingsCheckpoints"),
          h(
            "div",
            { class: "row" },
            folder,
            h(
              "button",
              {
                class: "nowrap",
                onclick: async () => {
                  const selected = await openDialog({ directory: true });
                  if (typeof selected === "string") {
                    setPref("checkpointDir", selected);
                    render();
                  }
                },
              },
              icon("folder"),
              t("exportChoose"),
            ),
          ),
        ),
        h("div", { class: "setting-row" }, h("span", null, t("settingsCache")), h("span", { class: "mono muted small ellipsis" }, info?.hf_cache ?? "")),
        h(
          "div",
          { class: "setting-row" },
          h("span", { class: "muted small" }, t("settingsTokenHint")),
          h(
            "button",
            {
              disabled: forgotten,
              onclick: async () => {
                await api.forgetToken();
                forgotten = true;
                render();
              },
            },
            t("settingsToken"),
          ),
        ),
      ),
      llamaBox,
      h(
        "section",
        { class: "card stack" },
        h("h3", null, t("settingsAbout")),
        h("div", { class: "setting-row" }, h("span", null, "Heretic"), h("span", { class: "num" }, `v${info?.version ?? ""}`)),
        h("div", { class: "setting-row" }, h("span", null, t("settingsCuda")), h("span", { class: "muted" }, info?.cuda_compiled ? t("yes") : t("no"))),
        updatesBox,
        h("p", { class: "faint small" }, "AGPL-3.0-or-later · heretic-project.org"),
      ),
    );
    renderUpdates();
    renderLlama();
  }

  // llama.cpp (GGUF quantization) lives in its own box so checks don't rebuild the page.
  const llamaBox = h("section", { class: "card stack" });
  const llamaInput = h("input", { type: "text", placeholder: "C:\\llama.cpp" });
  llamaInput.value = getLlamaCppDir();
  const llama = { checking: false, result: null };

  async function checkLlama() {
    const dir = llamaInput.value.trim();
    setLlamaCppDir(dir);
    llama.result = null;
    if (!dir) {
      renderLlama();
      return;
    }
    llama.checking = true;
    renderLlama();
    try {
      llama.result = await api.checkLlamaCpp(dir);
    } catch (error) {
      llama.result = { ok: false, error: String(error) };
    }
    llama.checking = false;
    renderLlama();
  }
  llamaInput.addEventListener("change", checkLlama);

  function llamaStatus() {
    if (llama.checking) return h("span", { class: "badge accent" }, t("updateChecking"));
    if (!llamaInput.value.trim()) return h("span", { class: "faint" }, t("settingsLlamaNotSet"));
    if (!llama.result) return null;
    if (llama.result.ok) {
      return h(
        "span",
        { class: "badge good" },
        llama.result.version ? t("settingsLlamaVersion", { version: llama.result.version }) : t("settingsLlamaFound"),
      );
    }
    return h("span", { class: "badge bad" }, t("settingsLlamaMissing"));
  }

  function renderLlama() {
    replace(
      llamaBox,
      h("h3", null, t("settingsLlama")),
      h(
        "label",
        { class: "field" },
        t("settingsLlamaFolder"),
        h(
          "div",
          { class: "row" },
          llamaInput,
          h(
            "button",
            {
              class: "nowrap",
              onclick: async () => {
                const selected = await openDialog({ directory: true });
                if (typeof selected === "string") {
                  llamaInput.value = selected;
                  checkLlama();
                }
              },
            },
            icon("folder"),
            t("exportChoose"),
          ),
        ),
      ),
      h(
        "div",
        { class: "setting-row" },
        llamaStatus() ?? h("span"),
        h(
          "span",
          { class: "row" },
          h("button", { class: "link", onclick: () => api.openReleasePage(LLAMA_CPP_RELEASES) }, t("settingsLlamaReleases")),
          h("button", { class: "nowrap", disabled: llama.checking, onclick: checkLlama }, icon("search"), t("settingsLlamaCheck")),
        ),
      ),
      llama.result && !llama.result.ok ? h("p", { class: "small bad-text" }, llama.result.error) : null,
      h("p", { class: "faint small" }, t("settingsLlamaHint")),
    );
  }
  if (llamaInput.value.trim()) checkLlama();

  // Updates live in their own box so checks and downloads don't rebuild the whole page.
  const updatesBox = h("div", { class: "updates stack" });

  const checkbox = (key, onchange) => {
    const input = h("input", { type: "checkbox" });
    input.checked = !!prefs[key];
    input.addEventListener("change", () => {
      setPref(key, input.checked);
      onchange?.();
    });
    return input;
  };

  function updateStatus() {
    const update = state.update;
    if (state.updateBusy === "checking") return h("span", { class: "badge accent" }, t("updateChecking"));
    if (state.updateError && !update) return h("span", { class: "badge bad" }, t("updateFailed"));
    if (update?.available) {
      return h(
        "span",
        { class: "row" },
        h("span", { class: "badge accent" }, t("updateAvailable")),
        h("span", { class: "num muted" }, `v${update.latest}`),
        update.prerelease ? h("span", { class: "faint small" }, t("updatePrerelease")) : null,
      );
    }
    if (update) return h("span", { class: "badge good" }, t("updateUpToDate"));
    return h("span", { class: "faint" }, t("updateNever"));
  }

  function renderUpdates() {
    const update = state.update;
    const progress = installProgress();
    const checked = prefs.updateLastCheck
      ? t("updateLastCheck", { time: new Date(prefs.updateLastCheck).toLocaleString(prefs.lang) })
      : null;
    replace(
      updatesBox,
      h("span", { class: "muted" }, t("updateSection")),
      h("label", { class: "setting-row" }, h("span", null, t("updateAuto")), checkbox("updateAuto")),
      h(
        "label",
        { class: "setting-row" },
        h("span", null, t("updatePrereleases")),
        checkbox("updatePrereleases", () => checkForUpdates()),
      ),
      h(
        "div",
        { class: "setting-row" },
        h("span", { class: "stack tight" }, updateStatus(), checked ? h("span", { class: "faint small" }, checked) : null),
        h(
          "button",
          { class: "nowrap", disabled: !!state.updateBusy, onclick: () => checkForUpdates() },
          icon("search"),
          t("updateCheck"),
        ),
      ),
      update?.available
        ? (progress ??
            h(
              "div",
              { class: "row wrap" },
              h("button", { class: "primary", onclick: installUpdate }, icon("download"), installLabel()),
              h("button", { class: "link", onclick: openReleaseNotes }, t("updateNotes")),
            ))
        : null,
      update?.available && update.install === "browser" && !progress
        ? h("p", { class: "faint small" }, t("updateManual"))
        : null,
      state.updateError ? h("p", { class: "small bad-text" }, state.updateError) : null,
    );
  }

  render();
  let lastInfo = state.info;
  let lastDark = isDark();
  let lastUpdate = [state.update, state.updateBusy, state.updateProgress, state.updateError];
  return subscribe(() => {
    if (state.info !== lastInfo || isDark() !== lastDark) {
      lastInfo = state.info;
      lastDark = isDark();
      render();
    }
    const current = [state.update, state.updateBusy, state.updateProgress, state.updateError];
    if (current.some((value, i) => value !== lastUpdate[i])) {
      lastUpdate = current;
      renderUpdates();
    }
  });
}
