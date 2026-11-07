// SPDX-License-Identifier: AGPL-3.0-or-later
// Export dialog: save locally or upload to Hugging Face.

import { api } from "../api.js";
import { h, progressBar, replace } from "../dom.js";
import { state, subscribe, t } from "../store.js";
import { openDialog } from "../tauri.js";
import { modal } from "../components/widgets.js";

export function openExport(mode) {
  const local = { tab: mode, strategy: "merge", user: null, card: null, working: false };
  const modelName = (state.settings?.model ?? "model").split(/[\\/]/).pop();

  const tabs = h("div", { class: "segmented" });
  const strategies = h("div", { class: "choices" });
  const content = h("div", { class: "stack" });
  const status = h("div", { class: "stack" });
  const footer = h("div", { class: "row modal-foot" });

  // Inputs are created once so typing isn't interrupted by re-renders.
  const directory = h("input", { type: "text" });
  const token = h("input", { type: "password", placeholder: "hf_..." });
  const repo = h("input", { type: "text", placeholder: "user/model-heretic" });
  const isPrivate = h("input", { type: "checkbox" });
  const cardPreview = h("pre", { class: "card-preview mono" }, "…");
  const details = h("details", null, h("summary", null, t("exportCard")), cardPreview);
  details.addEventListener("toggle", async () => {
    if (details.open && local.card === null) {
      try {
        local.card = await api.modelCardPreview();
      } catch (error) {
        local.card = String(error);
      }
      cardPreview.textContent = local.card;
    }
  });

  function showStatus(message, error) {
    replace(
      status,
      local.working && state.progress
        ? h("div", { class: "stack tight" }, h("span", { class: "muted small" }, state.progress.task), progressBar(state.progress.total ? state.progress.done / state.progress.total : null))
        : local.working
          ? progressBar(null)
          : null,
      message ? h("span", { class: "badge good" }, message) : null,
      error ? h("div", { class: "error-box" }, error) : null,
    );
  }

  async function checkUser(silent) {
    try {
      local.user = await api.hfWhoami(token.value || null);
      if (!repo.value) repo.value = `${local.user.name}/${modelName}-heretic`;
      showStatus(null, null);
    } catch (error) {
      local.user = null;
      if (!silent) showStatus(null, String(error));
    }
    render();
  }

  async function run(action, successMessage) {
    local.working = true;
    render();
    showStatus(null, null);
    try {
      const result = await action();
      local.working = false;
      showStatus(successMessage(result), null);
    } catch (error) {
      local.working = false;
      showStatus(null, String(error));
    }
    render();
  }

  function render() {
    replace(
      tabs,
      ["local", "hub"].map((tab) =>
        h(
          "button",
          {
            class: local.tab === tab ? "active" : "",
            onclick: () => {
              local.tab = tab;
              render();
              if (tab === "hub" && !local.user) checkUser(true);
            },
          },
          tab === "local" ? t("exportLocal") : t("exportHub"),
        ),
      ),
    );

    replace(
      strategies,
      [
        ["merge", "exportMerge", "exportMergeHint"],
        ["adapter", "exportAdapter", "exportAdapterHint"],
      ].map(([value, title, hint]) =>
        h(
          "button",
          {
            class: `choice card${local.strategy === value ? " active" : ""}`,
            onclick: () => {
              local.strategy = value;
              render();
            },
          },
          h("b", null, t(title)),
          h("span", { class: "muted small" }, t(hint)),
        ),
      ),
    );

    if (local.tab === "local") {
      replace(
        content,
        h(
          "label",
          { class: "field" },
          t("exportFolder"),
          h(
            "div",
            { class: "row" },
            directory,
            h(
              "button",
              {
                onclick: async () => {
                  const folder = await openDialog({ directory: true });
                  if (typeof folder === "string") directory.value = folder;
                },
              },
              t("exportChoose"),
            ),
          ),
        ),
      );
    } else {
      replace(
        content,
        local.user
          ? h("span", { class: "badge good" }, t("exportLoggedIn", { name: local.user.fullname ?? local.user.name }))
          : h(
              "label",
              { class: "field" },
              t("exportToken"),
              h("div", { class: "row" }, token, h("button", { onclick: () => checkUser(false) }, t("exportCheck"))),
              h("span", { class: "hint" }, t("exportTokenHint")),
            ),
        h("label", { class: "field" }, t("exportRepo"), repo),
        h("label", { class: "row" }, isPrivate, t("exportPrivate")),
        details,
      );
    }

    replace(
      footer,
      h("button", { onclick: () => close() }, t("close")),
      local.tab === "local"
        ? h(
            "button",
            {
              class: "primary",
              disabled: local.working,
              onclick: () => {
                if (!directory.value.trim()) return;
                run(
                  () => api.saveModel(directory.value.trim(), local.strategy),
                  (result) => t("exportSaved", { path: result.directory }),
                );
              },
            },
            t("exportSave"),
          )
        : h(
            "button",
            {
              class: "primary",
              disabled: local.working,
              onclick: () => {
                if (!repo.value.trim()) return;
                run(
                  () => api.uploadModel(token.value || null, repo.value.trim(), isPrivate.checked, local.strategy),
                  (result) => t("exportUploaded", { repo: result.repo_id }),
                );
              },
            },
            t("exportUpload"),
          ),
    );
  }

  const unsubscribe = subscribe(() => local.working && showStatus(null, null));
  const close = modal({
    title: t("actionSave"),
    width: "620px",
    body: h("div", { class: "stack" }, tabs, strategies, content, status, footer),
    onclose: unsubscribe,
  });

  render();
  if (mode === "hub") checkUser(true);
}
