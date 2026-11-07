// SPDX-License-Identifier: AGPL-3.0-or-later

import { api } from "../api.js";
import { h, replace } from "../dom.js";
import { icon } from "../icons.js";
import { isDark, prefs, setPref, state, subscribe, t } from "../store.js";
import { openDialog } from "../tauri.js";
import { DEFAULT_THEME, THEMES, themePreview } from "../themes.js";

/** A tile with a miniature of the theme in the current mode. */
function themeTile(theme, active, onpick) {
  const c = themePreview(theme.id, isDark());
  return h(
    "button",
    { class: `theme-tile${active ? " active" : ""}`, title: theme.name, "aria-pressed": String(active), onclick: onpick },
    h(
      "span",
      { class: "swatch", style: `background: ${c.bg}` },
      h(
        "span",
        { class: "swatch-card", style: `background: ${c.surface}; border-color: ${c.line}; border-radius: ${c.radius ?? "4px"}` },
        h("span", { class: "swatch-line", style: `background: ${c.text}` }),
        h("span", { class: "swatch-dot", style: `background: ${c.accent}` }),
      ),
      h(
        "span",
        { class: "swatch-card", style: `background: ${c.surface}; border-color: ${c.line}; border-radius: ${c.radius ?? "4px"}` },
        h("span", { class: "swatch-line", style: `background: ${c.muted}` }),
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
      h(
        "section",
        { class: "card stack" },
        h("h3", null, t("settingsAbout")),
        h("div", { class: "setting-row" }, h("span", null, "Heretic"), h("span", { class: "num" }, `v${info?.version ?? ""}`)),
        h("div", { class: "setting-row" }, h("span", null, t("settingsCuda")), h("span", { class: "muted" }, info?.cuda_compiled ? t("yes") : t("no"))),
        h("p", { class: "faint small" }, "AGPL-3.0-or-later · heretic-project.org"),
      ),
    );
  }

  render();
  let lastInfo = state.info;
  let lastDark = isDark();
  return subscribe(() => {
    if (state.info !== lastInfo || isDark() !== lastDark) {
      lastInfo = state.info;
      lastDark = isDark();
      render();
    }
  });
}
