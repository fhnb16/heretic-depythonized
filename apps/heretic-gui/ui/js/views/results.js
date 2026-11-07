// SPDX-License-Identifier: AGPL-3.0-or-later

import { api, paretoFront, plain } from "../api.js";
import { h, replace, rich } from "../dom.js";
import { icon } from "../icons.js";
import { isBusy, refreshTrials, set, state, subscribe, t } from "../store.js";
import { scatterPlot } from "../components/scatter.js";
import { openExport } from "./export.js";

export function mount(root) {
  const list = h("div", { class: "stack" });
  const plot = h("div", { class: "card stack" });
  const selectedPanel = h("div");
  const errorBox = h("div");
  const moreCount = h("input", { type: "number", min: 1, max: 1000, value: "20", style: "width: 96px", "aria-label": t("resultsMore") });
  const moreButton = h("button", null, icon("play"), t("resultsRun"));
  let applying = null;
  let renderedKey = "";

  moreButton.addEventListener("click", async () => {
    const count = Number(moreCount.value);
    if (!(count >= 1)) return;
    try {
      set({ total: state.trials.length + count, view: "run" });
      await api.runMoreTrials(count);
    } catch (error) {
      replace(errorBox, h("div", { class: "error-box" }, String(error)));
    }
  });

  root.append(
    h(
      "div",
      { class: "stack-lg" },
      h("header", { class: "page-head" }, h("div", { class: "stack tight" }, h("h1", null, t("resultsTitle")), h("p", { class: "muted" }, t("resultsHint")))),
      errorBox,
      h(
        "div",
        { class: "columns-2-3" },
        list,
        h(
          "div",
          { class: "stack" },
          plot,
          selectedPanel,
          h("div", { class: "card row" }, h("span", { class: "grow" }, t("resultsMore")), moreCount, moreButton),
        ),
      ),
    ),
  );

  async function apply(trial) {
    if (isBusy() || applying !== null || trial.number === state.appliedTrial) return;
    applying = trial.number;
    replace(errorBox);
    render(state, true);
    try {
      const result = await api.restoreTrial(trial.number);
      set({ appliedTrial: result.applied_trial });
    } catch (error) {
      replace(errorBox, h("div", { class: "error-box" }, String(error)));
    } finally {
      applying = null;
      render(state, true);
    }
  }

  function render(s, force = false) {
    const key = `${s.trials.length}:${s.appliedTrial}:${applying}:${s.status}`;
    if (!force && key === renderedKey) return;
    renderedKey = key;
    const busy = isBusy() || applying !== null;
    moreButton.disabled = busy;

    replace(
      list,
      paretoFront(s.trials).map((trial) =>
        h(
          "button",
          {
            class: `trial card${trial.number === s.appliedTrial ? " selected" : ""}`,
            disabled: busy && applying !== trial.number,
            onclick: () => apply(trial),
          },
          h(
            "div",
            { class: "row" },
            h("span", { class: "index" }, `#${trial.index}`),
            h("span", { class: "grow" }),
            applying === trial.number
              ? h("span", { class: "badge accent" }, t("resultsApplying"))
              : trial.number === s.appliedTrial
                ? h("span", { class: "badge accent" }, t("resultsSelected"))
                : null,
          ),
          h(
            "div",
            { class: "scores" },
            trial.scores.map((record) =>
              h(
                "div",
                { class: "score" },
                h("span", { class: "muted small" }, record.name),
                h("span", { class: "value" }, rich(record.score.display)),
                h("span", { class: "faint tiny" }, `${t("resultsBaseline")} `, h("span", { class: "mono" }, plain(record.baseline.display))),
              ),
            ),
          ),
        ),
      ),
    );

    replace(
      plot,
      h("div", { class: "row" }, h("h3", { class: "grow" }, t("resultsAllTrials")), h("span", { class: "faint small" }, t("resultsChartHint"))),
      scatterPlot({ trials: s.trials, names: s.objectiveNames, selected: s.appliedTrial, onselect: apply }),
    );

    const selected = s.trials.find((trial) => trial.number === s.appliedTrial);
    replace(
      selectedPanel,
      selected
        ? h(
            "div",
            { class: "card stack" },
            h(
              "div",
              { class: "row wrap" },
              h("button", { class: "primary", disabled: busy, onclick: () => set({ view: "chat" }) }, icon("message"), t("actionChat")),
              h("button", { disabled: busy, onclick: () => openExport("local") }, icon("download"), t("actionSave")),
              h("button", { disabled: busy, onclick: () => openExport("hub") }, icon("upload"), t("actionUpload")),
            ),
            h("h3", null, `${t("resultsParameters")} `, h("span", { class: "mono faint" }, `#${selected.index}`)),
            h(
              "table",
              null,
              h(
                "tbody",
                null,
                selected.parameters.map(([name, value]) => h("tr", null, h("td", { class: "muted mono small" }, name), h("td", { class: "num" }, value))),
              ),
            ),
          )
        : null,
    );
  }

  if (state.modelLoaded && state.trials.length === 0) refreshTrials();
  render(state, true);
  return subscribe((s) => render(s));
}
