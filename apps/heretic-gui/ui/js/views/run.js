// SPDX-License-Identifier: AGPL-3.0-or-later

import { api, formatDuration, plain } from "../api.js";
import { h, progressBar, replace } from "../dom.js";
import { icon } from "../icons.js";
import { isBusy, set, state, subscribe, t } from "../store.js";
import { scatterPlot } from "../components/scatter.js";
import { logPanel, stepper } from "../components/widgets.js";
import { explain } from "../components/help.js";

function statusLabel(status) {
  return {
    preparing: t("runPreparing"),
    running: t("runRunning"),
    failed: t("runFailed"),
  }[status] ?? t("runReady");
}

export function mount(root) {
  const header = h("header", { class: "page-head" });
  const progress = h("div", { class: "card stack" });
  const plot = h("div", { class: "card stack" });
  const side = h("div", { class: "card stack" });
  const columns = h("div", { class: "columns-3-2" }, plot, side);
  const log = logPanel();
  const logToggle = h("button", { class: "ghost toggle", "aria-expanded": "false" }, icon("chevron"), t("runLog"));
  let showLog = false;
  let stopping = false;
  let plotKey = "";

  logToggle.addEventListener("click", () => {
    showLog = !showLog;
    render(state);
  });

  root.append(
    h(
      "div",
      { class: "stack-lg" },
      header,
      progress,
      explain("helpRunTitle", "helpRunBody"),
      columns,
      h("div", { class: "card stack" }, logToggle, log.element),
    ),
  );

  function render(s) {
    const busy = isBusy();
    if (!busy) stopping = false;
    const done = s.trials.length;

    const meta = [h("span", { class: `badge ${s.status === "failed" ? "bad" : s.status === "ready" ? "good" : "accent"}` }, statusLabel(s.status))];
    if (s.status === "running" || s.status === "ready") {
      meta.push(h("span", { class: "muted small" }, `${t("runTrial")} `, h("span", { class: "num" }, `${done}/${s.total}`)));
      meta.push(h("span", { class: "muted small" }, `${t("runElapsed")} `, h("span", { class: "num" }, formatDuration(s.elapsed))));
      if (s.remaining != null && busy) {
        meta.push(h("span", { class: "muted small" }, `${t("runRemaining")} `, h("span", { class: "num" }, `~${formatDuration(s.remaining)}`)));
      }
    }

    const actions = [];
    if (busy) {
      actions.push(
        h(
          "button",
          {
            disabled: stopping,
            onclick: async () => {
              stopping = true;
              render(state);
              await api.stopRun();
            },
          },
          icon("pause"),
          t("runStop"),
        ),
        h("button", { class: "danger", onclick: () => api.cancelRun() }, icon("stop"), t("runCancel")),
      );
    } else if (s.status === "ready") {
      actions.push(h("button", { class: "primary big", onclick: () => set({ view: "results" }) }, t("runShowResults"), icon("right")));
    }

    replace(
      header,
      h("div", { class: "grow stack tight" }, h("h2", { class: "ellipsis mono" }, s.settings?.model ?? ""), h("div", { class: "run-meta" }, meta)),
      h("div", { class: "row" }, actions),
    );

    replace(
      progress,
      stepper(s.phase, s.status === "failed"),
      s.status === "running" && s.total > 0 ? progressBar(done / s.total) : null,
      s.progress
        ? h(
            "div",
            { class: "stack tight" },
            h("span", { class: "muted small" }, s.progress.task),
            progressBar(s.progress.total ? s.progress.done / s.progress.total : null),
          )
        : null,
      s.status === "failed" && s.statusMessage ? h("div", { class: "error-box" }, s.statusMessage) : null,
    );

    // Re-render the chart only when the trials change.
    const key = `${s.trials.length}:${s.appliedTrial}:${s.objectiveNames.join("|")}`;
    if (key !== plotKey) {
      plotKey = key;
      replace(
        plot,
        h("h3", null, t("resultsAllTrials")),
        s.trials.length
          ? scatterPlot({ trials: s.trials, names: s.objectiveNames, selected: s.appliedTrial })
          : h("p", { class: "muted placeholder" }, t("runNoTrials")),
      );
    }

    if (s.current && s.status === "running") {
      replace(
        side,
        h("h3", null, `${t("runTrial")} `, h("span", { class: "mono" }, s.current.index)),
        h("p", { class: "muted small" }, t("runCurrentParameters")),
        h("table", null, h("tbody", null, s.current.parameters.map(([name, value]) => h("tr", null, h("td", { class: "muted mono small" }, name), h("td", { class: "num" }, value))))),
      );
    } else if (s.trials.length) {
      const latest = [...s.trials].sort((a, b) => b.number - a.number).slice(0, 8);
      replace(
        side,
        h("h3", null, t("runLatest")),
        h(
          "table",
          null,
          h("thead", null, h("tr", null, h("th", null, "#"), latest[0].scores.map((score) => h("th", { class: "num" }, score.name)))),
          h(
            "tbody",
            null,
            latest.map((trial) =>
              h(
                "tr",
                null,
                h(
                  "td",
                  { class: "num" },
                  h("span", { class: "row" }, trial.index, trial.pareto_optimal ? h("span", { class: "pareto-mark", title: t("resultsParetoMark") }, icon("step", 14)) : null),
                ),
                trial.scores.map((score) => h("td", { class: "num" }, plain(score.score.display))),
              ),
            ),
          ),
        ),
      );
    }
    // Until the first trial there is nothing to show beside the chart; after a failure before
    // any trial, the chart itself is empty too.
    const empty = !s.trials.length && !(s.current && s.status === "running");
    side.hidden = empty;
    columns.classList.toggle("single", empty);
    columns.hidden = empty && s.status === "failed";

    const logVisible = showLog || s.status === "failed";
    logToggle.setAttribute("aria-expanded", String(logVisible));
    log.element.hidden = !logVisible;
    if (logVisible) log.update(s.logs, s.logGeneration);
  }

  render(state);
  return subscribe(render);
}
