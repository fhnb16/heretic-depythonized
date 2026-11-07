// SPDX-License-Identifier: AGPL-3.0-or-later
// Stepper, log panel and modal dialog.

import { h, replace, rich } from "../dom.js";
import { icon } from "../icons.js";
import { t } from "../store.js";

const STEPS = [
  ["loading_model", "phaseLoadingModel"],
  ["batch_size", "phaseBatchSize"],
  ["response_prefix", "phaseResponsePrefix"],
  ["scorers", "phaseScorers"],
  ["modifier", "phaseModifier"],
  ["optimizing", "phaseOptimizing"],
  ["finished", "phaseFinished"],
];

/** Progress through the phases of a run. */
export function stepper(phase, failed) {
  let current = phase == null ? -1 : STEPS.findIndex(([p]) => p === phase);
  // Phases after optimization (restoring, exporting) count as finished.
  if (phase != null && current === -1) current = STEPS.length - 1;
  const finished = phase === "finished" || current === STEPS.length - 1;

  return h(
    "ol",
    { class: "stepper" },
    STEPS.map(([, label], i) => {
      const done = i < current || (finished && i === current);
      const active = i === current && !finished;
      const classes = [done ? "done" : "", active ? "active" : "", failed && i === current ? "failed" : ""].join(" ");
      return h("li", { class: classes }, h("span", { class: "dot" }, done ? icon("check", 12) : i + 1), h("span", null, t(label)));
    }),
  );
}

/** A log panel that appends new lines incrementally. Returns { element, update(logs, generation) }. */
export function logPanel(height = "260px") {
  const element = h("div", { class: "log mono", style: `height: ${height}` });
  let rendered = 0;
  let generation = -1;
  let follow = true;

  element.addEventListener("scroll", () => {
    follow = element.scrollHeight - element.scrollTop - element.clientHeight < 24;
  });

  function update(logs, logGeneration) {
    if (logGeneration !== generation || rendered > logs.length) {
      replace(element);
      rendered = 0;
      generation = logGeneration;
    }
    const fragment = document.createDocumentFragment();
    for (; rendered < logs.length; rendered++) {
      const line = logs[rendered];
      fragment.append(h("div", { class: `line ${line.level}` }, rich(line.text)));
    }
    element.append(fragment);
    if (follow) element.scrollTop = element.scrollHeight;
  }

  return { element, update };
}

/**
 * Shows a modal dialog. `body` is a node; `buttons` is a list of nodes for the footer.
 * Returns a function that closes the dialog.
 */
export function modal({ title, body, buttons = [], width = "520px", onclose = null }) {
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    backdrop.remove();
    document.removeEventListener("keydown", onkey);
    onclose?.();
  };
  const onkey = (event) => event.key === "Escape" && close();

  const backdrop = h(
    "div",
    { class: "backdrop", onclick: (event) => event.target === backdrop && close() },
    h(
      "div",
      { class: "modal card", style: `width: ${width}`, role: "dialog", "aria-modal": "true" },
      h(
        "div",
        { class: "row" },
        h("h2", { class: "grow" }, title),
        h("button", { class: "ghost icon-only", "aria-label": t("close"), title: t("close"), onclick: close }, icon("x")),
      ),
      h("div", { class: "modal-body" }, body),
      buttons.length ? h("div", { class: "row modal-foot" }, buttons) : null,
    ),
  );

  document.addEventListener("keydown", onkey);
  document.body.append(backdrop);
  return close;
}
