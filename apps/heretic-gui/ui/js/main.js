// SPDX-License-Identifier: AGPL-3.0-or-later
// Application shell: sidebar navigation and view switching.

import { api } from "./api.js";
import { h, progressBar, replace } from "./dom.js";
import { icon } from "./icons.js";
import { applyTheme, initEvents, isBusy, launch, prefs, set, setPref, state, subscribe, t } from "./store.js";
import * as home from "./views/home.js";
import * as newRun from "./views/newrun.js";
import * as run from "./views/run.js";
import * as results from "./views/results.js";
import * as chat from "./views/chat.js";
import * as settings from "./views/settings.js";

const VIEWS = { home, new: newRun, run, results, chat, settings };

const NAV = [
  ["home", "home", "navHome", () => true],
  ["new", "plus", "navNewRun", () => !isBusy()],
  ["run", "activity", "navRun", () => state.status !== "idle"],
  ["results", "front", "navResults", () => state.modelLoaded && state.trials.length > 0],
  ["chat", "message", "navChat", () => state.modelLoaded && state.appliedTrial !== null && !isBusy()],
  ["settings", "sliders", "navSettings", () => true],
];

const nav = h("nav");
const statusBox = h("div");
const main = h("main");

document.getElementById("app").append(
  h(
    "div",
    { class: "layout" },
    h("aside", null, h("div", { class: "brand" }, icon("step", 20), "Heretic"), nav, h("span", { class: "grow" }), statusBox),
    main,
  ),
);

let mountedView = null;
let unmount = null;

function mountView(force = false) {
  if (!force && mountedView === state.view) return;
  unmount?.();
  const view = h("div", { class: "view" });
  replace(main, view);
  mountedView = state.view;
  unmount = (VIEWS[state.view] ?? home).mount(view) ?? null;
  main.scrollTop = 0;
}

function renderShell() {
  replace(
    nav,
    NAV.map(([view, name, label, enabled]) =>
      h(
        "button",
        { class: `nav${state.view === view ? " active" : ""}`, disabled: !enabled(), onclick: () => set({ view }) },
        icon(name),
        t(label),
      ),
    ),
  );

  if (state.status === "idle") {
    replace(statusBox);
    return;
  }
  const label = { preparing: t("runPreparing"), running: t("runRunning"), failed: t("runFailed") }[state.status] ?? t("runReady");
  const color = isBusy() ? "accent" : state.status === "failed" ? "bad" : "good";
  const running = state.status === "running";
  replace(
    statusBox,
    h(
      "button",
      { class: "status card", onclick: () => set({ view: "run" }) },
      h(
        "span",
        { class: "row" },
        h("span", { class: `badge ${color}` }, label),
        running ? h("span", { class: "num faint" }, `${state.trials.length}/${state.total}`) : null,
      ),
      running ? progressBar(state.total ? state.trials.length / state.total : 0) : null,
    ),
  );
}

subscribe(() => {
  renderShell();
  mountView();
});

// Changing the language rebuilds the current view, since labels are rendered once.
window.addEventListener("heretic:relabel", () => {
  renderShell();
  mountView(true);
});

async function start() {
  applyTheme();
  renderShell();
  mountView();
  try {
    await initEvents();
    const info = await api.appInfo();
    if (!prefs.checkpointDir) setPref("checkpointDir", info.default_checkpoint_dir);
    set({ info });
  } catch (error) {
    console.error(error);
  }
}

// Handle for debugging and UI tests (ui-test/).
window.heretic = { state, set, launch };

start();
