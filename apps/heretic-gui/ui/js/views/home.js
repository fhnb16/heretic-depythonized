// SPDX-License-Identifier: AGPL-3.0-or-later

import { api } from "../api.js";
import { h, progressBar, replace } from "../dom.js";
import { icon } from "../icons.js";
import { isBusy, launch, prefs, set, subscribe, t } from "../store.js";

function date(seconds) {
  if (seconds == null) return "";
  return new Date(seconds * 1000).toLocaleString(prefs.lang === "ru" ? "ru-RU" : "en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function mount(root) {
  const list = h("div", { class: "grid" });
  let projects = [];
  let loadedFor = null;

  root.append(
    h(
      "div",
      { class: "stack-lg" },
      h(
        "header",
        { class: "page-head" },
        h("div", { class: "stack" }, h("h1", null, t("homeTitle")), h("p", { class: "muted" }, t("homeSubtitle"))),
        h("button", { class: "primary big", onclick: () => set({ view: "new" }) }, icon("plus"), t("homeStart")),
      ),
      h("section", { class: "stack" }, h("h2", null, t("homeProjects")), list),
    ),
  );

  function render() {
    if (projects.length === 0) {
      replace(list, h("p", { class: "muted empty card" }, t("homeNoProjects")));
      return;
    }
    replace(
      list,
      projects.map((project) =>
        h(
          "div",
          { class: "card project" },
          h(
            "div",
            { class: "row" },
            h("h3", { class: "grow ellipsis", title: project.model }, project.model || "?"),
            project.finished
              ? h("span", { class: "badge good" }, t("projectFinished"))
              : h("span", { class: "badge warn" }, t("projectInterrupted")),
          ),
          progressBar(project.n_trials ? project.completed_trials / project.n_trials : 0),
          h(
            "div",
            { class: "row" },
            h("span", { class: "num muted" }, `${project.completed_trials}${project.n_trials ? `/${project.n_trials}` : ""}`),
            h("span", { class: "muted small" }, t("projectTrials")),
            h("span", { class: "grow" }),
            h("span", { class: "faint small" }, date(project.modified)),
          ),
          h(
            "div",
            { class: "row" },
            h(
              "button",
              {
                class: "primary",
                disabled: isBusy() || !project.settings,
                onclick: () => launch(project.settings, "continue"),
              },
              project.finished ? t("projectResults") : t("projectContinue"),
            ),
            h("span", { class: "grow" }),
            h(
              "button",
              {
                class: "ghost danger icon-only",
                title: t("projectDelete"),
                "aria-label": t("projectDelete"),
                onclick: async () => {
                  if (!confirm(t("projectDeleteConfirm"))) return;
                  await api.deleteProject(project.path);
                  await refresh();
                },
              },
              icon("trash"),
            ),
          ),
        ),
      ),
    );
  }

  async function refresh() {
    loadedFor = prefs.checkpointDir;
    try {
      projects = prefs.checkpointDir ? await api.listProjects(prefs.checkpointDir) : [];
    } catch {
      projects = [];
    }
    render();
  }

  refresh();
  return subscribe(() => {
    if (prefs.checkpointDir !== loadedFor) refresh();
  });
}
